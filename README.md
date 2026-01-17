# Week 5 Project – Arabic NLP CLI Tool

## Project Structure

```
nlp-cli-tool
│
├── main.py                 # Entry point for the CLI
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── commands/               # CLI commands
│   ├── generate.py         # Synthetic data generation
│   ├── eda.py              # Exploratory Data Analysis
│   ├── preprocessing.py   # Text preprocessing
│   ├── embedding.py       # Embedding generation
│   └── training.py        # Model training
│
├── utils/                  # Helper utilities
│   ├── arabic_text.py      # Arabic text normalization & cleaning
│   ├── data_handler.py     # CSV / data loading utilities
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py   # Plots & charts
│
├── outputs/
│   ├── data/               # Generated & cleaned CSV files
│   ├── embeddings/         # Serialized embeddings (.pkl)
│   ├── models/             # Trained models
│   ├── reports/            # Metrics & reports
│   └── visualizations/     # Saved plots
│
└── resources/
    ├── list.txt
    └── stopwords_nltk.txt
```

---

## 1. Environment Setup

### Create virtual environment (Python 3.11)

```bash
cd week_5_project_nlp
uv venv -p 3.11
```

### Activate the environment

**Mac / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

---

## 2. Synthetic Data Generation

This step is responsible for **generating synthetic Arabic text data** for classification using Gemini.

* `--topic` : You can choose any topic or domain you want
* `--count` : Number of rows (samples) to generate
* `--class` : Classification labels (repeat this option for multiple classes)
* `--api-key` : Gemini API key (the model is free to use)

Example:

```bash
uv run python main.py generate generate --topic car --count 100 --class mad --class happy --class normal --api-key YOUR_API_KEY --head 5
```

Notes:

* To add multiple classes, simply repeat `--class`
* Output is a CSV file with `text,label`

---

## 3. Exploratory Data Analysis (EDA)

### Class Distribution

**Pie chart**

```bash
uv run python main.py eda distribution --csv-path outputs/data/synthetic.csv --label-col label --plot-type pie
```

**Bar chart**

```bash
uv run python main.py eda distribution --csv-path outputs/data/synthetic.csv --label-col label --plot-type bar
```

---

### Text Length Histograms

**Word-level histogram**

```bash
uv run python main.py eda histogram --csv-path outputs/data/synthetic.csv --text-col text --plot-type word --bins 100
```

**Character-level histogram**

```bash
uv run python main.py eda histogram --csv-path outputs/data/synthetic.csv --text-col text --plot-type char --bins 100
```

---

## 4. Text Preprocessing

This stage is dedicated to Arabic text preprocessing and normalization.

Important behavior:

* **Any option not written will NOT be executed**
* **Once an option is written, it is automatically set to `True`**

You do NOT need to write `True/False`.
Just include the option name if you want to apply it.

Available operations:

* `--remove`
* `--replace-light`
* `--replace-aggressive`
* `--stopwords`

Example:

```bash
uv run python main.py text preprocess --csv-path outputs/data/output.csv --text-col text --output cl1 --remove --replace-light --replace-aggressive --stopwords
```

---

## 5. Embedding Generation

Create vector embeddings using **Model2Vec**.

```bash
uv run python main.py embd model2vec \
  --csv-path outputs/data/cl1.csv \
  --text-col text \
  --label-col label \
  --output hi3
```

**Output:**

* Serialized embeddings file (`hi3.pkl`)

---

## 6. Model Training

This stage is responsible for training machine learning models.

Multiple models are supported. You can:

* Train a single model
* Or train all supported models at once

⚠️ Important note:

* When using `--models all`, **hyperparameters cannot be customized** in this version.

### Supported Models & Tunable Hyperparameters

Each model accepts its standard sklearn hyperparameters **when used individually**.

* **KNN (KNeighborsClassifier)**

  * `n_neighbors`
  * `weights`
  * `metric`

* **Logistic Regression** (`lr`, `logistic`, `regression`)

  * `C`
  * `penalty`
  * `solver`
  * `class_weight`

* **Random Forest** (`rf`, `random`, `forest`)

  * `n_estimators`
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`

Example (train all models):

```bash
uv run python main.py model train --data-path outputs/embeddings/hi3.pkl --models all --output-col label --input-col text --test-size 0.2 --save-model allFirst.pkl
```

---

## 7. Full Pipeline Summary

1. Create environment & install dependencies
2. Generate synthetic Arabic text data
3. Explore data distribution & text statistics
4. Preprocess and normalize Arabic text
5. Generate embeddings
6. Train and save ML models

---

## Notes

* Paths assume project root execution
* Replace `YOUR_API_KEY` with a valid Gemini API key
* Designed for extensibility (new EDA, embeddings, models)

---

✅ This README documents the **end‑to‑end NLP CLI workflow** from raw text generation to trained models.
