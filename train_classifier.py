import os
import joblib
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTChar, LTFigure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import re


def overlaps(line_bbox, box_bbox, margin=2):
    lx0, ly0, lx1, ly1 = line_bbox
    bx0, by0, bx1, by1 = box_bbox
    return not (lx1 < bx0 - margin or lx0 > bx1 + margin or ly1 < by0 - margin or ly0 > by1 + margin)


HEADING_CUES = [
    "introduction", "background", "conclusion", "summary", "abstract", "overview",
    "references", "discussion", "results", "acknowledgments", "appendix", "table of contents"
]

def has_semantic_cue(text):
    lower_t = text.strip().lower()
    return int(any(cue in lower_t for cue in HEADING_CUES))

def extract_title_lines(layout_items):
    page1_items = [item for item in layout_items if item["page"] == 1]
    if not page1_items:
        return []
    max_size = max(item["font_size"] for item in page1_items)
    top_y0_threshold = max(item["y0"] for item in page1_items) * 0.65

    return [
        item["text"].strip()
        for item in page1_items
        if abs(item["font_size"] - max_size) < 0.2 and item["y0"] > top_y0_threshold
    ]

def extract_features_from_pdf(pdf_path):
    features, labels = [], []
    layout_items, all_font_sizes = [], set()
    headings_by_page, box_regions = {}, {}

    try:
        for page_num, layout in enumerate(extract_pages(pdf_path)):
            page_height = layout.bbox[3]
            headings_by_page[page_num] = []
            box_regions[page_num] = []

            for element in layout:
                if isinstance(element, (LTTextBox, LTFigure)):
                    box_regions[page_num].append((element.x0, element.y0, element.x1, element.y1))

            for element in layout:
                if isinstance(element, LTTextBox):
                    for line in element:
                        chars = [char for char in line if isinstance(char, LTChar)]
                        if not chars:
                            continue

                        text = line.get_text().strip()
                        if not text or len(text) < 3:
                            continue

                        avg_size = np.mean([c.size for c in chars])
                        all_font_sizes.add(round(avg_size, 1))

                        is_bold = any("Bold" in c.fontname or "Black" in c.fontname for c in chars)
                        starts_with_number = text[:3].strip()[0].isdigit() if text else False

                        x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
                        box_overlap = any(overlaps((x0, y0, x1, y1), b) for b in box_regions[page_num])
                        word_count = len(text.split())
                        proximity_to_top = y0 / page_height
                        match = re.match(r'^(\d+(\.\d+)*)(\s+|[\)\.])', text)
                        depth = match.group(1).count('.') + 1 if match else 0

                        layout_items.append({
                            "text": text,
                            "font_size": avg_size,
                            "is_bold": is_bold,
                            "starts_with_number": starts_with_number,
                            "y0": y0,
                            "page": page_num,
                            "word_count": word_count,
                            "depth": depth,
                            "proximity_to_top": proximity_to_top,
                            "box_overlap": int(box_overlap),
                            "indentation": x0,
                            "semantic_cue": has_semantic_cue(text),
                        })
    except Exception as e:
        print(f"❌ Skipping {pdf_path}: {e}")
        return [], []

    title_lines = extract_title_lines(layout_items)
    uniform_font = len(all_font_sizes) == 1

    previous_label = None

    for item in layout_items:
        label = None
        if item["text"].strip() in title_lines and item["page"] == 1:
            continue

        if not uniform_font:
            if item["font_size"] > 14 and item["word_count"] <= 12 and item["proximity_to_top"] > 0.5:
                label = "H1"
            elif item["font_size"] > 13 and item["word_count"] <= 15:
                label = "H2"
            elif item["font_size"] > 12:
                label = "H3"
        else:
            if item["starts_with_number"] and item["indentation"] < 100:
                if item["depth"] == 1:
                    label = "H1"
                elif item["depth"] == 2:
                    label = "H2"
                else:
                    label = "H3"
            elif not item["starts_with_number"]:
                if item["indentation"] < 100:
                    label = "H1"
                elif item["indentation"] < 150:
                    label = "H2"
                else:
                    label = "H3"
            elif item["starts_with_number"] and item["indentation"] >= 100:
                if previous_label == "H1":
                    label = "H2"
                elif previous_label == "H2":
                    label = "H3"

        if label == "H3":
            if not item["is_bold"]:
                continue
            if ":" in item["text"]:
                colon_index = item["text"].index(":")
                after_colon = item["text"][colon_index + 1:].strip()
                if len(after_colon.split()) <= 3:
                    item["text"] = item["text"][:colon_index].strip()

        vertical_order = 0.0
        prev = headings_by_page[item["page"]]
        if label:
            if prev:
                vertical_order = abs(item["y0"] - prev[-1])
            prev.append(item["y0"])
            previous_label = label

            features.append([
                item["font_size"], int(item["is_bold"]), int(item["starts_with_number"]), item["y0"],
                item["depth"], item["word_count"], item["proximity_to_top"], item["box_overlap"],
                item["indentation"], item["semantic_cue"], vertical_order
            ])
            labels.append(label)

    print(f"✅ Labeled: {len(labels)} lines in {os.path.basename(pdf_path)}")
    return features, labels

def extract_features_from_folder(folder_path):
    all_features, all_labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"🔍 Extracting: {filename}")
            f, l = extract_features_from_pdf(os.path.join(folder_path, filename))
            all_features.extend(f)
            all_labels.extend(l)
    return np.array(all_features), all_labels

def train_model(pdf_folder):
    X, y = extract_features_from_folder(pdf_folder)
    if len(X) == 0:
        print("⚠️ No valid training data found. Exiting.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    print(f"✅ Trained. Accuracy: {clf.score(X_test, y_test):.2f}")
    print(f"🔧 Best Parameters: {grid_search.best_params_}")

    os.makedirs("app", exist_ok=True)
    joblib.dump((clf, le), "model.pkl")
    print("📦 Saved model to app/model.pkl")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_classifier.py <folder_with_pdfs>")
    else:
        train_model(sys.argv[1])