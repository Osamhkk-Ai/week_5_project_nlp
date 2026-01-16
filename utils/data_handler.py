from google import genai
import pandas as pd
from pathlib import Path
from utils.arabic_text import remove_text , normalize_arabic_letters , normalize_whitespace , remove_stopwords , aggressive_normalize

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



def clean_arabic_text(text: str, * , remove: bool = False, replace_light: bool = False, replace_aggressive: bool = False, stopwords: bool = False):
    """
    Arabic text cleaning orchestrator.
    """
    if not isinstance(text, str):
        return text

    if remove:
        text = remove_text(text)

    if replace_light:
        text = normalize_arabic_letters(text)

    if replace_aggressive:
        text = aggressive_normalize(text)

    if stopwords:
        text = remove_stopwords(text)
    text = normalize_whitespace(text)
    return text


