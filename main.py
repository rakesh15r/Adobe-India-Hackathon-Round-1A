import os
import json
from utils import detect_headings

INPUT_DIR = "app/input"
OUTPUT_DIR = "app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(INPUT_DIR, filename)
    print(f"📄 Processing: {pdf_path}")
    
    title, outline = detect_headings(pdf_path)

    output = {
        "title": title,
        "outline": outline
    }

    out_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {out_path}")
