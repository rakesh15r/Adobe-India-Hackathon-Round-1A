Certainly! Here is a **README.md** for your Round 1A "Connecting the Dots" Hackathon solution, reflecting your approach, training data, code modules, and instructions:

# Adobe Hackathon 2025: Connecting the Dots  
## Round 1A — Structured PDF Outline Extraction

> **Challenge**: Rethink how machines understand PDFs by extracting document structure (Title, H1, H2, H3) for smarter reading and research applications.

## 🧠 Solution Overview

This project automates extraction of hierarchical outlines (title, headings H1/H2/H3 + page number) from PDF files. The solution is based on a robust ML approach using features derived from PDF layout and typography, and is tailored for speed and generalizability across diverse document styles.

**Key Points:**
- No hard-coded heading logic, works on unseen PDF formats.
- Trained on 1000 real-world research PDFs for robustness.
- Handles outline detection with high precision and recall.
- Fast, offline, dockerized CPU solution within size/time limits.

Certainly! Here’s a **README.md** combining your detailed structure, the specifics of your solution, and best practices aligned to the Adobe Hackathon Round 1A requirements. This synthesizes all your technical elements, makes installation and usage clear, and is hackathon presentation-ready.


## 🚀 Objective

Extracts structured and hierarchical outlines from PDF documents—including **title**, as well as **H1**, **H2**, and **H3** headings—with high precision and recall. Outputs are compliant with the hackathon's required JSON format.

## 📂 Example Output

```json
{
  "title": "Document Title Here",
  "outline": [
    { "level": "H1", "text": "Main Heading", "page": 0 },
    { "level": "H2", "text": "Sub Heading", "page": 1 },
    { "level": "H3", "text": "Sub-sub Heading", "page": 2 }
  ]
}
```

## 🧠 Solution Highlights

- **ML-based heading classification** (RandomForest) – not just font-size!
- **Rich, robust feature extraction** using `pdfminer.six` for layout, font, and structure.
- Semantic, positional, and style-based cues for generalizing to diverse PDFs.
- Handles multilingual PDFs (works with Unicode: Japanese, Hindi, English tested).
- Optimized for speed: 🖥️ CPU-only, 💾 <200MB model size, 📦 offline, ⏱️ <10s per 50-page PDF.
- Resilient to noisy OCR, boxed content, headers/footers, and more.
- Dockerized for hackathon portability and consistent execution.

## ⚙️ Project Structure

```
.
├── main.py                  # Entry: run PDF --> JSON outline
├── utils.py                 # Core feature extraction & model inference
├── train_classifier.py      # Build RandomForest model from labeled PDFs
├── model.pkl                # Trained model (RandomForest + LabelEncoder)
├── input/                   # Place input PDFs here (runtime)
├── output/                  # JSON outline results saved here
├── Dockerfile               # For offline, CPU-only docker execution
└── README.md
```

## 📚 How It Works

### 1. Training  
- Uses `train_classifier.py` to extract ∼10 features per line from 1000+ diverse arXiv PDFs (`input_arxiv/`).
- **Features include:** font size, boldness, numbering, indentation, depth, word count, overlap, and more.
- Trains a `RandomForestClassifier` to learn heading patterns, serializes as `model.pkl`.

### 2. Inference
- `main.py` processes PDFs from `input/`.
- Feature vectors are computed line-by-line (`utils.py`).
- Model predicts heading class for each line; non-heading lines filtered.
- Assembles hierarchical outline (Title, H1/H2/H3 with page numbers).
- Output JSON is saved in `output/` and matches hackathon format.



## 🐳 Run Offline & Locally with Docker

**1. Build Docker Image:**  
```sh
docker build -t pdf-outline-extractor .
```

**2. Place your PDFs:**  
Put all PDFs to process in the `input/` folder.

**3. Execute Container:**  
```sh
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  pdf-outline-extractor
```

A `<name>.json` file is created in `output/` for every `<name>.pdf` in `input/`.

- ⛔ Network disabled; CPU only; all code, data, and model is local/offline.
- ⏱️ Will process 50-page PDFs in well under 10 seconds.

## 🧪 Tested On

- Academic papers (arXiv), annual reports, textbooks, research PDFs.
- Hindi and Japanese PDFs (Unicode), English multi-style documents.
- Documents with boxed headings, bullets, rich formatting, TOC pages, etc.

## 📈 Model Feature Table

| Feature            | Description                                  |
|--------------------|----------------------------------------------|
| font_size          | Avg. font size in line                       |
| is_bold            | Bold font or not (binary)                    |
| starts_with_number | Numbering cue at line start                  |
| y0                 | Top-of-page position (vertical)              |
| depth              | Outline numbering depth (e.g., 2.3 = 2)      |
| word_count         | Word count in line                           |
| proximity_to_top   | Relative y0 position on page                 |
| box_overlap        | Overlap with figure/table box                |
| indentation        | x0 (left-space) position                     |
| vertical_order     | y0 delta to prev. heading on page            |

## 📝 Direct CLI Execution (for dev/test)

```sh
python main.py
```
Processes all `.pdf` files in `input/` folder and saves results to `output/`.

Certainly! Here’s the section in an **edit-README** format, using markdown and keeping edits clearly segmentable for drop-in to your main README.md file.  
You can copy-paste these blocks directly.

## 🔧 Dependencies

- **Python 3.8+**
- **pdfminer.six** (for PDF parsing)
- **numpy**
- **scikit-learn**
- **joblib**

All dependencies are installed within the Docker image automatically—no manual pip install is needed on your host.  
(For local development: `pip install -r requirements.txt`.)

## 📝 Solution Components

- **main.py** — Entry-point: orchestrates extraction for each PDF and JSON output.
- **utils.py** — PDF parsing, feature engineering, and ML-based heading detection/inference.
- **model-best.pkl** — Trained heading classifier (RandomForest + LabelEncoder, self-contained, <200MB).
- **train_classifier.py** — Training pipeline: generates and tunes the classifier from labeled PDFs (run outside Docker container).
- **input/** — Directory for input PDF files at runtime.
- **output/** — Directory for structured outline JSON output files.

## ⚡ Performance & Limits

- Model benchmarked on 50-page PDFs: completes within the 10 seconds runtime limit.
- Model binary (`model-best.pkl`) is <200MB; no network or internet needed; operates strictly on CPU (no GPU).
- Handles a broad variety of PDF structures and heading styles.
- Designed as modular and reusable code—ready for expansion in Round 1B and beyond.

## 💡 Improvements & Notes

- Delivers strong performance on arXiv and general research documents, as well as reports with varied formatting.
- Heading detection is **not solely** based on font size; it leverages positional, visual, and semantic features for robustness.
- Easily extensible for downstream document intelligence tasks (e.g., section extraction, summary, recommendation).
- Codebase structured to facilitate integration with future challenge rounds and real-world productionization.


## ✍️ Authors / Team

- Team: code OG
- Submission for **Adobe India Hackathon 2025 (Round 1A)**
- Challenge: "Connecting the Dots Through Docs"


