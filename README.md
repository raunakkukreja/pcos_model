# PCOSense — Model Training & Research (`pcos_model`)

Machine-learning code behind **[PCOSense](https://pcosense-app.vercel.app)**, a privacy-first, explainable web app for early PCOS (Polycystic Ovary Syndrome) risk screening.

This repository holds everything needed to **train, evaluate and explain** the two models that power the app. The web application itself (React frontend, FastAPI backend, Cloud Run CV worker) lives in the companion repo, **[PCOS_website](https://github.com/raunakkukreja/PCOS_website)**, which is also checked out under `website/` here.

> ⚠️ **Disclaimer:** PCOSense is a screening aid, not a diagnostic tool. Its output is not medical advice. Please see a doctor for any health concern.

---

## The two models

| | Tabular risk model | Ultrasound model |
|---|---|---|
| **Input** | 7-step survey: demographics, cycle, symptoms, lifestyle, optional labs (32 raw features + 2 engineered scores) | Ovary ultrasound image |
| **Algorithm** | XGBoost classifier | ResNet-18 (transfer learning, PyTorch) |
| **Imbalance handling** | `scale_pos_weight` | `WeightedRandomSampler` + class weights |
| **Explainability** | SHAP (`TreeExplainer`) — top risk drivers per prediction | CAM / Grad-CAM heatmap over the image |
| **Training data** | `PCOS DataSet.csv` — 684 rows (521 PCOS / 163 non-PCOS) | ~6.4k labelled images (`infected` / `noninfected`) |
| **Deployed in** | FastAPI backend on Render (`website/backend`) | Docker worker on Google Cloud Run (`website/cv_worker`) |

### Results

| Model | Evaluation | Result |
|---|---|---|
| XGBoost | 80/20 stratified split (137 test rows) | Accuracy ≈ 0.89–0.91, PR-AUC ≈ 0.99 |
| ResNet-18 v1 *(deployed)* | In-domain validation (1,289 images) | ≈ 99% accuracy |
| ResNet-18 v1 | Independent dataset `ultrasound_` (1,922 images) | 1,914 / 1,922 correct (0 false negatives, 8 false positives) |
| ResNet-18 v2 *(heavier regularisation, not deployed)* | In-domain / independent | 96.9% / 96.7% |

**Read these numbers with care.** The tabular set is small (684 rows, ~76% positive) and evaluated on a single split, so the confidence interval is wide. The CNN's training set is an aggregation of public ultrasound datasets and likely overlaps with the "independent" Kaggle test set, and splits are per image rather than per patient — so the cross-dataset score is probably optimistic. See `docs/PCOSense_System_Architecture_Deep_Dive.md` for the full analysis.

---

## Repository layout

```
PCOS/
├── models/
│   ├── main_model/                 # Tabular (XGBoost) pipeline
│   │   ├── data_cleaning.py        #   load + clean the survey CSV
│   │   ├── train_xgboost.py        #   feature engineering, training, evaluation
│   │   ├── explain_shap.py         #   SHAP summary + waterfall plots
│   │   ├── demo_predict.py         #   interactive CLI demo
│   │   └── xgb_model.pkl           #   trained model (git-ignored)
│   ├── train_cnn.py                # ResNet-18 v1 training (deployed model)
│   ├── train_cnn_improved_v2.py    # v2 with stronger regularisation (experiment)
│   ├── test_on_ultrasound.py       # cross-dataset evaluation of v1
│   ├── test_v2_on_ultrasound.py    # cross-dataset evaluation of v2
│   ├── compare_datasets.py         # compatibility check between the two image sets
│   ├── explain_ultrasound.py       # Grad-CAM heatmap for a single image
│   ├── predict_ultrasound_demo.py  # inference + heatmap + confidence demo
│   ├── generate_report_graphs.py   # ROC / PR / confusion-matrix figures for the report
│   ├── generate_text_metrics.py    # printable metrics for the report
│   ├── requirements.txt            # tabular dependencies
│   ├── requirements_cv.txt         # computer-vision dependencies
│   └── artifacts*/                 # weights, plots, training history (git-ignored)
├── website/                        # → PCOS_website repo (React + FastAPI + CV worker)
├── docs/                           # project report, architecture deep-dive, viva notes
├── PROJECT_SUMMARY.md              # high-level system summary
└── notes.txt                       # personal command cheat-sheet
```

---

## Datasets

Datasets and trained weights are **not committed** (see `models/.gitignore`). Obtain them and place them as shown:

| Purpose | Location | Source |
|---|---|---|
| Survey data (tabular) | `models/PCOS DataSet.csv` | [Kaggle – mdabdulkaium/pcos-dataset](https://www.kaggle.com/datasets/mdabdulkaium/pcos-dataset) (684 rows) |
| CNN training images | `models/ultrasound_alisha/PCOS Data/{infected,noninfected}` | [Kaggle – alisha786/pcos-ultrasound-dataset](https://www.kaggle.com/datasets/alisha786/pcos-ultrasound-dataset), a de-duplicated copy of the [figshare dataset](https://doi.org/10.6084/m9.figshare.27682557) |
| Independent CNN test images | `models/ultrasound_/data/{train,test}/{infected,notinfected}` | [Kaggle – anaghachoudhari/pcos-detection-using-ultrasound-images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images) |

Please respect each dataset's licence and terms.

---

## Getting started

Requires **Python 3.11+**. A CUDA GPU is strongly recommended for CNN training (developed on an RTX 4050); inference runs fine on CPU.

```bash
# 1. Create an environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)

# 2. Install what you need
pip install -r models/requirements.txt        # tabular model
pip install -r models/requirements_cv.txt     # ultrasound model (installs PyTorch; for CUDA use the matching wheel from pytorch.org)
```

> **Heads-up:** several scripts have absolute Windows paths hard-coded in a `CONFIG` block near the top (e.g. `c:\coding\my projects\PCOS\models\...`). Update those paths for your machine before running.

### Train and explain the tabular model

```bash
cd models/main_model
python data_cleaning.py      # sanity-check the cleaned data
python train_xgboost.py      # trains, prints metrics, saves xgb_model.pkl
python explain_shap.py       # writes SHAP plots to artifacts/
python demo_predict.py       # answer a few questions, get a risk estimate
```

### Train and evaluate the ultrasound model

```bash
cd models
python train_cnn.py                  # saves artifacts/ultrasound_model.pth + curves + confusion matrix
python test_on_ultrasound.py         # evaluate on the independent dataset
python explain_ultrasound.py --image path/to/ultrasound.jpg   # heatmap for one image
```

`train_cnn_improved_v2.py` / `test_v2_on_ultrasound.py` reproduce the regularisation experiment; v2 generalised slightly worse, so v1 is the one shipped.

---

## From research to production

1. Train → `models/main_model/xgb_model.pkl` is copied to `website/backend/xgb_model.pkl`.
2. Train → `models/artifacts/ultrasound_model.pth` is copied to `website/cv_worker/model/ultrasound_model.pth`.
3. Push the website repo; Vercel (frontend), Render (API) and Cloud Run (CV worker) take it from there. See the [PCOS_website README](https://github.com/raunakkukreja/PCOS_website#readme).

The production backend does the SHAP explanation at request time and the CV worker computes a lightweight, no-gradient CAM to fit within free-tier memory limits; the Grad-CAM scripts here are for offline exploration.

## Further reading

- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — system overview
- [`docs/PCOSense_System_Architecture_Deep_Dive.md`](docs/PCOSense_System_Architecture_Deep_Dive.md) — architecture, metrics, limitations
- [`docs/mini_project_report.md`](docs/mini_project_report.md) — full project report

## Known limitations

- Self-reported survey inputs introduce response bias; the tabular dataset comes from a limited population and may not generalise.
- Small tabular dataset and single-split evaluation; no k-fold cross-validation yet.
- CNN evaluation may be affected by dataset overlap (see above). A patient-level split and a truly external test set are the next steps.
- `demo_ultrasound.py` at the repo root is legacy: it imports an inference module that has since moved to `website/cv_worker`.
