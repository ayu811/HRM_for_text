# save_wikitext_arrow.py （一度だけ実行）
from datasets import load_dataset
from pathlib import Path

subset = "wikitext-103-v1"  # 他: wikitext-2-v1, wikitext-103-raw-v1, wikitext-2-raw-v1
out_dir = Path("./wikitext_arrow") / subset

ds = load_dataset("Salesforce/wikitext", subset)  # train/validation/test を全部
out_dir.parent.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(out_dir))
print(f"[OK] Saved Arrow to: {out_dir}")

