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

## 📂 Repo Structure

```
.
├── main.py              # Entry point: PDF reading and JSON output
├── utils.py             # Feature extraction and heading detection logic (model inference)
├── train_classifier.py  # Training script (RandomForest) using labeled data
├── model-best.pkl       # Trained ML model (RandomForest, <200MB)
├── Dockerfile
├── input/               # (Runtime) Mountpoint for PDFs to process
├── output/              # (Runtime) Mountpoint for resulting JSON files
└── input_arxiv/         # PDF training set (not required for runtime)
```

## 🏋️ Training & Approach

### Dataset

- Used **arxiv-metadata-oai-snapshot** to select 1000 diverse research papers.
- Downloaded the PDFs (see `download.py`), and placed in `input_arxiv/`.

### Features

- Extracted per-line features: font size, boldness, indentation, numbers, vertical position, proximity to top, word count, outline cues, detected depth, etc.
- Title detection leverages size, top position, centering.
- Heading levels inferred via model: combines style and structural clues (not just font-size).

### Model

- **Classifier**: RandomForest (scikit-learn), tuned with GridSearchCV.
- Three label classes: H1, H2, H3 (plus background/non-heading lines discarded).
- Model and label encoder serialized as `model-best.pkl` (<200MB).

### Outline Extraction

- For each PDF:
  - Parse all lines and features (`utils.py`).
  - Predict heading classes via model.
  - Assemble hierarchical outline:
    - Title (from first page, large and centered lines)
    - H1/H2/H3 (with page number and text)

### Why ML?

- PDF layouts are inconsistent: headings are not always largest or boldest.
- Model learns patterns robust across document templates, fonts, and languages.

## 🚀 Running the Solution

Built to run **via Docker** as required by the challenge.  
**NOTE:** No network access, pure CPU, model size < 200MB, ≤10s/50p PDF.

### **Step 1: Build Docker Image**
```sh
docker build --platform linux/amd64 -t pdf_outline_extractor:latest .
```

### **Step 2: Run Extraction**
- Place your PDFs in a folder, e.g., `input/`
- Outputs (JSON) will appear in `output/`

```sh
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf_outline_extractor:latest
```

Each `filename.pdf` in `input/` produces `filename.json` in `output/` (see challenge format).

## 🗂️ Output Format

Example (as per hackathon):

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## 🔧 Dependencies

- **Python 3.8+**
- **pdfminer.six** (PDF parsing)
- **numpy, scikit-learn, joblib**
- All are installed inside Docker automatically.

## 📝 Solution Components

- **main.py** — Orchestrates extraction for each PDF.
- **utils.py** — PDF parsing, feature engineering, ML inference.
- **model-best.pkl** — Trained heading classifier (self-contained).
- **train_classifier.py** — Model training pipeline (run outside of container).

## ⚡ Performance & Limits

- Model tested on 50-page PDFs: completes within 10 seconds.
- Model binary < 200MB, no network, no GPU needed.
- Ready for diverse heading styles; processing is modular for Round 1B.

## 💡 Improvements & Notes

- Good performance on Arxiv/academic papers and general reports.
- Not solely font-size based; uses positional, typographical, and semantic cues.
- Extensible for further downstream document intelligence tasks.

### Contact or Questions?

Reach out for clarifications or collaboration!

**© 2025** — Hackathon submission, keep repository private until submission instructions specify otherwise.

---

Sources
[1] 6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc_copy.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85220937/ca180f98-8128-419d-b361-dd5c21bb62b3/6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc_copy.pdf
[2] download.py https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85220937/50fb8ed3-8b27-42cd-bb9c-2184d0f357a5/download.py
[3] train_classifier.py https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85220937/15946d0f-a3e9-45ec-8325-cbe0e96200bb/train_classifier.py
[4] main.py https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85220937/145cc0ae-5af8-467a-be1c-53ed0eb7b049/main.py
[5] utils.py https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85220937/75295909-9c62-4693-b239-cf1eb866d65f/utils.py
