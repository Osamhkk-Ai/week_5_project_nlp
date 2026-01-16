import re
from pyarabic.araby import strip_tashkeel, strip_tatweel
from camel_tools.utils.normalize import normalize_alef_ar, normalize_alef_maksura_ar
from pathlib import Path





def remove_links(text: str) -> str:
    """
    Remove URLs (http, https, www) from text.
    """
    if not isinstance(text, str):
        return text

    url_pattern = r"http\S+|www\S+"
    return re.sub(url_pattern, "", text)


def remove_tashkeel(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.
    """
    if not isinstance(text, str):
        return text

    return strip_tashkeel(text)


def remove_tatweel(text: str) -> str:
    """
    Remove Arabic tatweel (elongation character ـ).
    """
    if not isinstance(text, str):
        return text

    return strip_tatweel(text)


def normalize_arabic_letters(text: str) -> str:
    """
    Normalize Arabic letters:
    - Alef variants (أ إ آ → ا)
    - Alef maksura (ى → ي)
    """
    if not isinstance(text, str):
        return text

    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove Arabic and English punctuation from text.
    """
    if not isinstance(text, str):
        return text

    return re.sub(r"[^\w\s\u0600-\u06FF]", "", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by:
    - Removing leading/trailing spaces
    - Replacing multiple spaces with a single space
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# def clean_arabic_text(text: str) -> str:
#     """
#     Full Arabic text cleaning pipeline.
#     """
#     text = remove_links(text)
#     text = remove_tashkeel(text)
#     text = remove_tatweel(text)
#     text = normalize_arabic_letters(text)
#     text = remove_punctuation(text)
#     text = normalize_whitespace(text)
#     return text

def remove_text(text: str) -> str:
    """
    Remove unwanted elements from Arabic text:
    - links
    - tashkeel
    - tatweel
    - punctuation
    - extra whitespace
    """
    if not isinstance(text, str):
        return text

    text = remove_links(text)
    text = remove_tashkeel(text)
    text = remove_tatweel(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    return text



STOPWORDS_PATH = Path("resources/stopwords.txt")

def load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }

AR_STOPWORDS = load_stopwords()


def remove_stopwords(text: str) -> str:
    """
    Remove Arabic stopwords from text.
    """
    if not isinstance(text, str):
        return text

    words = text.split()
    filtered = [w for w in words if w not in AR_STOPWORDS]
    return " ".join(filtered)

def aggressive_normalize(text: str) -> str:
    """
    Aggressive Arabic normalization.
    WARNING: Use only for search / IR tasks.
    """

    if not isinstance(text, str):
        return text

    return (
        text
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
