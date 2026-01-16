import typer
from utils.data_handler import generate_classification_csv_gemini
import pandas as pd
generate_app = typer.Typer(help="Data generation commands" , invoke_without_command=True)


@generate_app.command()
def generate(
    topic: str = typer.Option("customer reviews",help="Subject/domain of the generated texts"),
    count: int = typer.Option(..., help="Number of rows to generate"),
    classes: list[str] = typer.Option(...,"--class", help="Repeat this option to add multiple classes. Example: --class positive --class negative"),
    api_key: str = typer.Option(..., help="Your Gemini API key (free tier)"),
    preview: int = typer.Option(5,"--head",help="Show N sample rows after generation",),
):
    """
    Generate synthetic Arabic classification data using Gemini.
    Output is saved automatically to outputs/output_data/synthetic.csv
    """
    if not api_key:
        raise ValueError("Please provide your own Gemini API key (free tier).")
    

    generate_classification_csv_gemini(
        topic=topic,
        num_rows=count,
        classes=classes,
        api_key=api_key,
    )
    print("✅ Synthetic data generated successfully.")
    df = pd.read_csv(r"C:\Users\HP\Desktop\SDAIA_BOOTCAMP5\nlp-cli-tool\outputs\data\synthetic.csv")
    print("\n📊 Sample (first 5 rows):")
    print(df.head(preview))