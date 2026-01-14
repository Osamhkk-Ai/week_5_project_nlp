from google import genai


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
