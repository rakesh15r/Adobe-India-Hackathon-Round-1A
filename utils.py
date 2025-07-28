import numpy as np
import joblib
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTChar, LTImage, LTFigure
import string
BULLET_CHARS = {'•', '‣', '∙', '*'}


def clean_heading_text(text):
    # Remove leading bullets and excessive whitespace
    text = text.lstrip(''.join(BULLET_CHARS)).strip()
    return text

# Load trained classifier and label encoder
clf, le = joblib.load("model.pkl")

def overlaps(line_bbox, box_bbox, margin=2):
    lx0, ly0, lx1, ly1 = line_bbox
    bx0, by0, bx1, by1 = box_bbox
    return not (lx1 < bx0 - margin or lx0 > bx1 + margin or ly1 < by0 - margin or ly0 > by1 + margin)

def extract_title_lines(layout_items):
    page1_items = [item for item in layout_items if item["page"] == 1]
    if not page1_items:
        return []

    max_size = max(item["font_size"] for item in page1_items)
    top_y_threshold = max(item["y0"] for item in page1_items) * 0.65
    center_min_x = 0.25
    center_max_x = 0.75

    title_lines = []
    prev_y0 = None

    # Sort top to bottom
    for item in sorted(page1_items, key=lambda x: -x["y0"]):
        if abs(item["font_size"] - max_size) > 1.0:
            continue
        if item["y0"] < top_y_threshold:
            continue
        x_center_ratio = (item.get("x0", 0) + item.get("x1", 0)) / 2 / 612
        if x_center_ratio < center_min_x or x_center_ratio > center_max_x:
            continue
        if prev_y0 is not None and abs(prev_y0 - item["y0"]) > 50:
            break  # likely a separate block

        title_lines.append(item["text"].strip())
        prev_y0 = item["y0"]

    return title_lines

def extract_layout_with_features(pdf_path):
    layout_items = []
    features = []
    box_regions = {}

    for page_num, layout in enumerate(extract_pages(pdf_path)):
        box_regions[page_num] = []
        page_height = layout.bbox[3]

        for el in layout:
            if isinstance(el, (LTTextBox, LTFigure)):
                box_regions[page_num].append((el.x0, el.y0, el.x1, el.y1))
        prev_y = None

        for element in layout:
            if isinstance(element, LTTextBox):
                for line in element:
                    chars = [char for char in line if isinstance(char, LTChar)]
                    if not chars:
                        continue

                    text = line.get_text().strip()
                    if not text or len(text) < 3:
                        continue
                    if re.fullmatch(r"[\d\s\.\(\)\%\+\-\/]+", text):
                        continue  # 🚫 Skip pure numeric/statistical lines

                    font_size = np.mean([c.size for c in chars])
                    is_bold = any("Bold" in c.fontname or "Black" in c.fontname for c in chars)
                    starts_with_number = text[:3].strip()[0].isdigit() if text else False
                    x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
                    line_bbox = (x0, y0, x1, y1)
                    box_overlap = any(overlaps(line_bbox, box) for box in box_regions[page_num])
                    word_count = len(text.split()) 
                    proximity_to_top = y0 / page_height
                    
                    if font_size < 8.5 and word_count <= 3 and box_overlap:
                        continue
                    if prev_y is not None and abs(prev_y - y0) < 5:
                        continue  # skip tightly stacked lines (common in tables)
                    prev_y = y0

                    match = re.match(r'^(\d+(\.\d+)*)(\s+|[\)\.])', text)
                    depth = match.group(1).count('.') + 1 if match else 0

                    layout_items.append({
                        "text": text,
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "starts_with_number": starts_with_number,
                        "y0": y0,
                        "x0": x0,
                        "x1": x1,
                        "page": page_num + 1,
                        "word_count": word_count,
                        "depth": depth,
                        "proximity_to_top": proximity_to_top,
                        "box_overlap": int(box_overlap)
                    })

    title_lines = extract_title_lines(layout_items)

    filtered_layout = []
    for item in layout_items:
        if item["text"].strip() in title_lines and item["page"] == 1:
            continue
        filtered_layout.append(item)
        features.append([
            item["font_size"],
            int(item["is_bold"]),
            int(item["starts_with_number"]),
            item["y0"],
            item["depth"],
            item["word_count"],
            item["proximity_to_top"],
            item["box_overlap"]
        ])

    return title_lines, filtered_layout, np.array(features)

def detect_headings(pdf_path):
    title_lines, layout_data, features = extract_layout_with_features(pdf_path)
    title = " ".join(title_lines).strip() if title_lines else ""

    if len(features) == 0:
        return title, []

    preds = clf.predict(features)
    levels = le.inverse_transform(preds)

    outline = []
    for item, lvl in zip(layout_data, levels):
        text = item["text"]

        # ❗ Filter out weak or non-heading lines
        if len(text) < 4:
            continue
        if re.fullmatch(r"[\d\s\.\(\)\%\+\-\/]+", text):
            continue  # ❗ Skip numeric/statistical-only lines
        if item["font_size"] < 10:
            continue
        if not item["is_bold"] and not item["starts_with_number"]:
            continue
        
        if ':' in text:
            colon_index = text.index(':')
            if colon_index < len(text) - 2:  # ensure there's meaningful text after
                text = text[:colon_index].strip()
        
        text = clean_heading_text(text)
        
        outline.append({
            "level": lvl,
            "text": text,
            "page": item["page"]-1
        })

    return title, outline
