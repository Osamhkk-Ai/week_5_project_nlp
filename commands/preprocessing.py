import pandas as pd
import typer
from utils.data_handler import clean_text_column, text_stats
from pathlib import Path
preprocess_app = typer.Typer()
OUTPUT_DIR = Path("outputs/data")
@preprocess_app.command()

def preprocess(
    csv_path: str = typer.Option(..., "--csv-path", help="Path to input CSV"),
    text_col: str = typer.Option(..., "--text-col", help="Text column to clean"),
    output: str = typer.Option("cleaned.csv", "--output", help="Output file name"),
    remove: bool = typer.Option(False, "--remove"),
    replace_light: bool = typer.Option(False, "--replace-light"),
    replace_aggressive: bool = typer.Option(False, "--replace-aggressive"),
    stopwords: bool = typer.Option(False, "--stopwords"),
):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise typer.BadParameter(f"Column '{text_col}' not found in CSV")

    # عشان لو ما حط اسم الصيغه  يعني 
    if not output.lower().endswith(".csv"):
        output += ".csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / output

    before = text_stats(df[text_col])

    df = clean_text_column(
        df,
        text_col=text_col,
        remove=remove,
        replace_light=replace_light,
        replace_aggressive=replace_aggressive,
        stopwords=stopwords,
    )

    after = text_stats(df[text_col])

    df.to_csv(output_path, index=False)

    typer.echo(" Cleaning completed")
    typer.echo(f" Rows: {before['rows']}")
    typer.echo(f" Avg chars: {before['avg_chars']:.2f} → {after['avg_chars']:.2f}")
    typer.echo(f" Total words: {before['total_words']} → {after['total_words']}")
