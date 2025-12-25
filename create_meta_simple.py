# create_meta_simple.py
# Safely build checkpoints/meta.json from data/labels.csv using only stdlib.
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "labels.csv"
OUT = ROOT / "checkpoints" / "meta.json"

if not MANIFEST.exists():
    raise SystemExit(f"ERROR: Manifest not found: {MANIFEST}. Create data/labels.csv first.")

# Read labels column and build mapping
labels = []
with MANIFEST.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    if 'label' not in reader.fieldnames:
        raise SystemExit("ERROR: 'label' column not found in data/labels.csv")
    for row in reader:
        lab = row['label'].strip()
        if lab:
            labels.append(lab)

labels = sorted(set(labels))
label_map = {lab: i for i, lab in enumerate(labels)}

meta = {
    "label_map": label_map,
    "input_dim": 2 * 21 * 3
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("✅ Wrot e meta.json with", len(label_map), "labels to", OUT)
