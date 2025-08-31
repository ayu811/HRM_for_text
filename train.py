#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM-LLM: 汎用 事前学習スクリプト（Arrow/DDP/MPS/CUDA対応、ステップ保存・自動再開付き）

- 事前に datasets.save_to_disk() 済みの Arrow データを読み込む（デフォ: WikiText）。
- "text" など任意の列名から tiktoken でトークナイズ→固定長(BLOCK_SIZE)に整形。
- 事前学習中は q_head（ハルティングヘッド）を凍結。
- CUDA: AMP/torch.compile を有効化（MPS/CPUは自動で無効化）。
- 途中落ち対策: エポック中も N ステップごとに checkpoint を保存・最新シンボリック更新。
- --resume or RESUME=1 で latest.pt を自動ロード（optimizer/scheduler/scalerも復元）。
- torchrun でそのまま DDP 動作（Gradient Accumulation / no_sync 対応）。

使い方例:
    torchrun --nproc_per_node=4 pretrain_generic.py \\
      --data_dir ./wikitext_arrow/wikitext-103-v1 \\
      --text_column text --block_size 256 --batch_size 8 \\
      --epochs 1 --save_every_steps 1000 --eval_every_steps 2000 --resume

環境変数でも上書き可能（CLIが優先）。
"""
import os
import math
import time
import json
import signal
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm.auto import tqdm

# optional deps
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    import wandb
except Exception:
    wandb = None

import tiktoken

# model import (プロジェクトルート直下の model.py を想定)
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import HRMConfig, HRMModel  # noqa

# ---------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------
def env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).lower() not in {"0", "false", "no"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_distributed() -> Tuple[int, int, int, torch.device]:
    """torchrun 前提を含めた DDP/MPS/CUDA/CPU 初期化"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device

def safe_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)

def save_checkpoint(
    save_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    world_size: int,
    rank: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if rank != 0:
        return
    state = {
        "step": step,
        "model": (model.module.state_dict() if world_size > 1 else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
        "saved_at": time.time(),
        "torch": torch.__version__,
    }
    # ステップ固有 + latest
    tagged = save_dir / f"step-{step:08d}.pt"
    latest = save_dir / "latest.pt"
    safe_save(state, tagged)
    safe_save(state, latest)

def load_checkpoint(latest_path: Path) -> Optional[Dict[str, Any]]:
    if latest_path.exists():
        return torch.load(latest_path, map_location="cpu")
    return None

# ---------------------------------------------------------
# データセット（Arrow + tiktoken 固定長）
# ---------------------------------------------------------
def build_tokenized_arrow(
    data_dir: Path,
    text_column: str,
    block_size: int,
    tokenizer_name: str = "gpt2",
) -> Tuple[Dataset, Optional[Dataset]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found. Please prepare Arrow via datasets.save_to_disk().")

    ds: DatasetDict = load_from_disk(str(data_dir))
    if text_column is None:
        # もっとも一般的そうな列名を推測
        for cand in ["text", "content", "document", "raw", "sentence"]:
            if cand in ds["train"].column_names:
                text_column = cand
                break
        else:
            raise ValueError(f"No text column found. Available columns: {ds['train'].column_names}")

    enc = tiktoken.get_encoding(tokenizer_name)

    def tokenize(batch):
        texts = batch[text_column]
        # 重要：長さを維持する（None/空でも要素は残す）
        ids = [enc.encode(t) if isinstance(t, str) and len(t) > 0 else [] for t in texts]
        return {"ids": ids}

    def group_texts(batch):
        all_ids = []
        for seq in batch["ids"]:
            all_ids.extend(seq)
        chunk_len = block_size + 1
        if len(all_ids) < chunk_len:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        # オーバーラップなしの純分割（エポック/シャッフルでカバー）
        chunks = [all_ids[i:i + chunk_len] for i in range(0, len(all_ids) - chunk_len + 1, chunk_len)]
        input_ids = [c[:-1] for c in chunks]
        labels = [c[1:] for c in chunks]
        attn = [[1] * block_size for _ in input_ids]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    processed = {}
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        tmp = ds[split]
        # 既存列は text_column だけ残しつつ ids へ
        cols_to_drop = [c for c in tmp.column_names if c != text_column]
        tmp = tmp.map(tokenize, batched=True, remove_columns=cols_to_drop, load_from_cache_file=False)
        tmp = tmp.map(group_texts, batched=True, remove_columns=[text_column, "ids"], load_from_cache_file=False)
        tmp = tmp.with_format("torch")
        processed[split] = tmp

    if "train" not in processed:
        raise ValueError("train split is required in Arrow dataset.")
    return processed["train"], processed.get("validation", None)

# ---------------------------------------------------------
# 引数/環境変数
# ---------------------------------------------------------
def build_args():
    import argparse
    p = argparse.ArgumentParser(description="Generic HRM-LLM Pretraining (Arrow)")
    p.add_argument("--data_dir", type=str, default=os.environ.get("WIKITEXT_DIR", "./wikitext_arrow/wikitext-103-v1"))
    p.add_argument("--text_column", type=str, default=os.environ.get("TEXT_COLUMN", "text"))
    p.add_argument("--tokenizer_name", type=str, default=os.environ.get("TOKENIZER_NAME", "gpt2"))

    p.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", 1)))
    p.add_argument("--block_size", type=int, default=int(os.environ.get("BLOCK_SIZE", 256)))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 8)))
    p.add_argument("--grad_accum_steps", type=int, default=int(os.environ.get("GRAD_ACCUM_STEPS", 2)))
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", 1)))

    p.add_argument("--lr_max", type=float, default=float(os.environ.get("LEARNING_RATE_MAX", 2e-4)))
    p.add_argument("--lr_min", type=float, default=float(os.environ.get("LEARNING_RATE_MIN", 1e-5)))
    p.add_argument("--weight_decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 0.01)))

    p.add_argument("--save_dir", type=str, default=os.environ.get("SAVE_DIR", "./checkpoints"))
    p.add_argument("--save_every_steps", type=int, default=int(os.environ.get("SAVE_EVERY_STEPS", 2000)))
    p.add_argument("--eval_every_steps", type=int, default=int(os.environ.get("EVAL_EVERY_STEPS", 4000)))
    p.add_argument("--resume", action="store_true", default=env_flag("RESUME", False))

    p.add_argument("--use_compile", action="store_true", default=env_flag("USE_COMPILE", True))
    p.add_argument("--mixed_precision", action="store_true", default=env_flag("MIXED_PRECISION", True))

    # HRM config（環境変数でも上書き可）
    p.add_argument("--d_model", type=int, default=int(os.environ.get("D_MODEL", 768)))
    p.add_argument("--n_heads", type=int, default=int(os.environ.get("N_HEADS", 12)))
    p.add_argument("--d_ff", type=int, default=int(os.environ.get("D_FF", 3072)))
    p.add_argument("--n_layers_h", type=int, default=int(os.environ.get("N_LAYERS_H", 3)))
    p.add_argument("--n_layers_l", type=int, default=int(os.environ.get("N_LAYERS_L", 6)))
    p.add_argument("--n_latents_l", type=int, default=int(os.environ.get("N_LATENTS_L", 32)))
    p.add_argument("--h_cycles", type=int, default=int(os.environ.get("H_CYCLES", 1)))
    p.add_argument("--l_cycles", type=int, default=int(os.environ.get("L_CYCLES", 1)))
    p.add_argument("--dropout", type=float, default=float(os.environ.get("DROPOUT", 0.1)))
    p.add_argument("--segments", type=int, default=int(os.environ.get("SEGMENTS", 2)))
    p.add_argument("--pad_id", type=int, default=int(os.environ.get("PAD_ID", 0)))
    p.add_argument("--tie_lm_head", action="store_true", default=env_flag("TIE_LM_HEAD", True))
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "HRM-LLM"))
    p.add_argument("--wandb_run", type=str, default=os.environ.get("WANDB_RUN_NAME", "pre_train"))
    p.add_argument("--wandb_disabled", action="store_true", default=env_flag("WANDB_DISABLED", False))
    return p.parse_args()

# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
def main():
    args = build_args()
    save_dir = Path(args.save_dir)

    set_seed(42)

    # TF32（CUDAのみ）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    rank, world_size, local_rank, device = init_distributed()
    IS_CUDA = (device.type == "cuda")
    IS_MPS = (device.type == "mps")

    # compile / AMP
    USE_COMPILE = IS_CUDA and args.use_compile
    USE_AMP = IS_CUDA and args.mixed_precision

    # Dataset/DataLoader
    train_ds, val_ds = build_tokenized_arrow(
        data_dir=Path(args.data_dir),
        text_column=args.text_column,
        block_size=args.block_size,
        tokenizer_name=args.tokenizer_name,
    )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    pin_memory = IS_CUDA  # MPS/CPU は False 推奨

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=(0 if IS_MPS else args.num_workers),
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=(0 if IS_MPS else args.num_workers),
            pin_memory=pin_memory,
        )

    # モデル設定
    vocab_size = tiktoken.get_encoding(args.tokenizer_name).n_vocab
    cfg = HRMConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads_h=args.n_heads,
        n_heads_l=args.n_heads,
        n_layers_h=args.n_layers_h,
        n_layers_l=args.n_layers_l,
        n_latents_l=args.n_latents_l,
        ffn_mult=args.d_ff / args.d_model,
        dropout=args.dropout,
        attn_dropout=0.0,
        max_seq_len=args.block_size,
        N_cycles=args.h_cycles,
        T_steps=args.l_cycles,
        segments=args.segments,
        tie_lm_head=args.tie_lm_head,
        pad_id=args.pad_id,
    )
    model = HRMModel(cfg).to(device)

    # ハルティングヘッドを固定
    if hasattr(model, "q_head"):
        for p in model.q_head.parameters():
            p.requires_grad = False

    # torch.compile（CUDAのみ）
    if USE_COMPILE:
        try:
            model = torch.compile(model)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] torch.compile disabled: {e}")

    # DDP
    if world_size > 1:
        if IS_CUDA:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    # Optimizer / Scheduler / AMP
    params = list(model.module.parameters() if world_size > 1 else model.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr_max, weight_decay=args.weight_decay)

    # ステップ数見積り（Cosine は総ステップ数が必要）
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr_min)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    # wandb
    if (rank == 0) and (wandb is not None) and (not args.wandb_disabled):
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        except Exception as e:
            print(f"[WARN] wandb.init failed: {e}")

    # 途中落ち対策：SIGTERM/CTRL-C で最後に保存
    interrupted = {"flag": False}
    def _handler(signum, frame):
        interrupted["flag"] = True
        print(f"\n[Rank {rank}] Caught signal {signum}. Will save checkpoint at next safe point...")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # 自動再開
    global_step = 0
    if args.resume:
        latest = save_dir / "latest.pt"
        ckpt = load_checkpoint(latest)
        if ckpt is not None:
            target = model.module if world_size > 1 else model
            target.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            global_step = ckpt.get("step", 0)
            if rank == 0:
                print(f"[RESUME] Loaded checkpoint at step={global_step}")

    # ------------------------ Training Loop ------------------------
    for epoch in range(args.epochs):
        model.train()
        if world_size > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        progress = train_loader if rank != 0 else tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):
            # 既に再開済みのステップをスキップ（DataLoader は消費される点に注意）
            # 大まかに同程度の合計ステップへ追いつくイメージで、精密な再現性は狙わない
            if global_step // args.grad_accum_steps > (epoch * steps_per_epoch + step) // args.grad_accum_steps:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=IS_CUDA)
            attention_mask = batch["attention_mask"].to(device, non_blocking=IS_CUDA)
            labels = input_ids

            is_accum = ((global_step + 1) % args.grad_accum_steps != 0)
            sync_ctx = (model.no_sync if (world_size > 1 and is_accum) else nullcontext)

            with sync_ctx():
                autocast_ctx = torch.amp.autocast('cuda') if USE_AMP else nullcontext()
                with autocast_ctx:
                    out = model(input_ids, labels=labels, attention_mask=attention_mask)
                    loss = out["loss"]

                loss = loss / args.grad_accum_steps
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if not is_accum:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # ログ
            if rank == 0:
                if (not is_accum) and (wandb is not None) and (not args.wandb_disabled):
                    try:
                        wandb.log({"train/loss": loss.item() * args.grad_accum_steps,
                                   "train/lr": scheduler.get_last_lr()[0],
                                   "train/step": global_step + 1})
                    except Exception:
                        pass
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix({"loss": f"{loss.item()*args.grad_accum_steps:.4f}",
                                          "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                                          "gstep": global_step+1})

            global_step += 1

            # ステップ保存
            if (global_step % args.save_every_steps == 0) or interrupted["flag"]:
                save_checkpoint(
                    save_dir=save_dir,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    world_size=world_size,
                    rank=rank,
                    extra={"epoch": epoch},
                )
                if interrupted["flag"]:
                    if rank == 0:
                        print("[INFO] Interrupted. Saved checkpoint. Exiting gracefully.")
                    if dist.is_initialized():
                        dist.barrier()
                        dist.destroy_process_group()
                    return

            # ステップ評価
            if (val_loader is not None) and (global_step % args.eval_every_steps == 0):
                if world_size > 1:
                    dist.barrier()
                if rank == 0:
                    eval_model = model.module if world_size > 1 else model
                    eval_model.eval()
                    tot = 0.0
                    with torch.no_grad():
                        for vb in val_loader:
                            vi = vb["input_ids"].to(device, non_blocking=IS_CUDA)
                            vm = vb["attention_mask"].to(device, non_blocking=IS_CUDA)
                            ctx = torch.amp.autocast('cuda') if USE_AMP else nullcontext()
                            with ctx:
                                out = eval_model(vi, labels=vi, attention_mask=vm)
                            tot += out["loss"].item()
                    avg = tot / max(1, len(val_loader))
                    ppl = math.exp(avg)
                    print(f"[VAL] step {global_step}: loss={avg:.4f} | ppl={ppl:.2f}")
                    if (wandb is not None) and (not args.wandb_disabled):
                        try:
                            wandb.log({"val/loss": avg, "val/ppl": ppl, "val/step": global_step})
                        except Exception:
                            pass
                    eval_model.train()
                if world_size > 1:
                    dist.barrier()

        # エポック終端でも保存（保険）
        save_checkpoint(
            save_dir=save_dir,
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            world_size=world_size,
            rank=rank,
            extra={"epoch": epoch, "epoch_end": True},
        )
        if world_size > 1:
            dist.barrier()

    # 最終保存 & 終了
    if rank == 0:
        latest = save_dir / "final.pt"
        safe_save(torch.load(save_dir / "latest.pt", map_location="cpu"), latest)
        print(f"[OK] Training complete. Final checkpoint: {latest}")
        if (wandb is not None) and (not args.wandb_disabled):
            try:
                wandb.finish()
            except Exception:
                pass

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

