from google import genai
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

from utils.arabic_text import clean_arabic_text
from sklearn.model_selection import train_test_split


def generate_classification_csv_gemini(
    num_rows: int,
    classes: list[str],
    topic: str,
    api_key: str,
    output_path: str = "outputs/data/synthetic.csv",
):
    client = genai.Client(api_key=api_key)

    prompt = f"""
Generate {num_rows} Arabic text classification samples.

The texts should be about: {topic}

Output MUST be valid CSV only.
Do NOT add explanations.
Do NOT add markdown.
Do NOT add extra text.

CSV format exactly:
text,label

Use ONLY these labels:
{", ".join(classes)}
"""


    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    csv_text = response.text.strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_text)



def load_class_distribution(csv_path: str, label_col: str,):
    """
    Load a CSV file and return class distribution.

    Returns:
        dict: {class_name: count}
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    class_counts = df[label_col].value_counts().to_dict()

    return class_counts


def load_text_length(csv_path: str,text_col: str):
    """
    Load a CSV file and return text length histogram.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    texts = df[text_col].dropna().astype(str).str.strip()

    text_counts_word = [len(t.split()) for t in texts]
    text_counts_char = [len(t) for t in texts]
    return text_counts_word , text_counts_char




def clean_text_column(df: pd.DataFrame,text_col: str,*,remove: bool = False,replace_light: bool = False,replace_aggressive: bool = False,stopwords: bool = False):
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found")

    df[text_col] = df[text_col].apply(
        lambda x: clean_arabic_text(
            x,
            remove=remove,
            replace_light=replace_light,
            replace_aggressive=replace_aggressive,
            stopwords=stopwords,
        )
    )
    return df



def text_stats(series: pd.Series) -> dict:
    texts = series.dropna().astype(str)

    return {
        "rows": len(texts),
        "avg_chars": texts.str.len().mean(),
        "total_words": texts.str.split().str.len().sum(),
    }






def load_data(
    data_path: str,
    input_col: str | None,
    output_col: str,
):
    """
    Load data from CSV or PKL.
    Returns X, y
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    # PKL
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)

        X = np.array(data["X"])
        y = np.array(data["y"])

    # CSV
    elif path.suffix == ".csv":
        if input_col is None:
            raise ValueError("input_col is required for CSV")

        df = pd.read_csv(path)

        X = df[input_col].tolist()
        y = df[output_col].values

        # لو embeddings كنص
        if isinstance(X[0], str):
            X = [eval(x) for x in X]

        X = np.array(X)

    else:
        raise ValueError("Only CSV or PKL supported")

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Split data into train/test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
