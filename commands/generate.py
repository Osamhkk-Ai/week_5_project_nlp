from utils.data_handler import generate_classification_csv_gemini


def generate_command(
    topic : str,
    count: int,
    classes: list[str],
    api_key: str,
):
    if not api_key:
        raise ValueError(
            "Please provide your own Gemini API key (free tier)."
        )

    generate_classification_csv_gemini(
        topic = topic,
        num_rows=count,
        classes=classes,
        api_key=api_key,
    )