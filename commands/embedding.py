from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import typer


embedding_app = typer.Typer()
DEFAULT_EMBEDDINGS_DIR = Path(
    r"C:\Users\HP\Desktop\SDAIA_BOOTCAMP5\nlp-cli-tool\outputs\embeddings"
)
@embedding_app.command()
def tfidf(
    csv_path: Path = typer.Option(..., help="Path to cleaned CSV file"),
    text_col: str = typer.Option(..., help="Text column name"),
    label_col: str = typer.Option(..., help="Label column name"),
    max_features: int = typer.Option(5000, help="Maximum TF-IDF features"),
    ngram_min: int = typer.Option(1, help="Minimum n-gram size"),
    ngram_max: int = typer.Option(1, help="Maximum n-gram size"),
    output: Path = typer.Option(..., help="Output pickle file path"),
):
    """
    Generate TF-IDF embeddings from text data.
    """

    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].values


    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
    )
    vectors = vectorizer.fit_transform(texts)

    memory_mb = (
        vectors.data.nbytes +
        vectors.indptr.nbytes +
        vectors.indices.nbytes
    ) / (1024 ** 2)

    if output.parent == Path("."):
        output = DEFAULT_EMBEDDINGS_DIR / output

    if output.suffix != ".pkl":
        output = output.with_suffix(".pkl")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"X": vectors,"y": labels},f
)


    typer.echo("\nTF-IDF Embedding Created Successfully")
    typer.echo("-----------------------------------")
    typer.echo(f"Documents      : {vectors.shape[0]}")
    typer.echo(f"Max Features   : {vectors.shape[1]}")
    typer.echo(f"N-gram Range   : ({ngram_min}, {ngram_max})")
    typer.echo(f"Shape          : {vectors.shape}")
    typer.echo(f"Memory Usage   : {memory_mb:.2f} MB")
    typer.echo(f"Saved To       : {output}\n")



@embedding_app.command()
def model2vec(
    csv_path: Path = typer.Option(..., help="Path to cleaned CSV file"),
    text_col: str = typer.Option(..., help="Text column name"),
    label_col: str = typer.Option(..., help="Label column name"),
    output: Path = typer.Option(
        ...,
        help="Output file name or path (.pkl will be added automatically)",
    ),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
):
    """
    Generate Model2Vec embeddings (static sentence embeddings).
    """

    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].values


    if output.parent == Path("."):
        output = DEFAULT_EMBEDDINGS_DIR / output

    if output.suffix != ".pkl":
        output = output.with_suffix(".pkl")

    output.parent.mkdir(parents=True, exist_ok=True)

    model_name = "JadwalAlmaa/model2vec-ARBERTv2"
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    memory_mb = embeddings.nbytes / (1024 ** 2)

    with open(output, "wb") as f:
        pickle.dump(
    {
        "X": embeddings,
        "y": labels
    },
    f
)


    typer.echo("\nModel2Vec Embedding Created Successfully")
    typer.echo("---------------------------------------")
    typer.echo(f"Documents    : {embeddings.shape[0]}")
    typer.echo(f"Vector Size  : {embeddings.shape[1]}")
    typer.echo(f"Shape        : {embeddings.shape}")
    typer.echo(f"Memory Usage : {memory_mb:.2f} MB")
    typer.echo(f"Saved To     : {output}\n")
