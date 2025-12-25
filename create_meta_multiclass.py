# create_meta_multiclass.py
import json
from pathlib import Path

DATA_DIR = Path("data/web_signs")
OUT_CSV = Path("data/labels.csv")
OUT_META = Path("checkpoints/meta_multiclass.json")

if not DATA_DIR.exists():
    raise SystemExit("❌ data/web_signs folder not found")

classes = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])

if not classes:
    raise SystemExit("❌ No class folders inside data/web_signs")

# Write labels.csv
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", encoding="utf-8") as f:
    f.write("label\n")
    for c in classes:
        f.write(c + "\n")

# Write meta JSON
OUT_META.parent.mkdir(parents=True, exist_ok=True)
with OUT_META.open("w", encoding="utf-8") as f:
    json.dump({"classes": classes}, f, indent=2)

print("\n✅ labels.csv and meta_multiclass.json created successfully!")
print("Classes:", classes)
