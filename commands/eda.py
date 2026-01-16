import typer
from utils.data_handler import load_class_distribution
from utils.visualization import (
    plot_class_distribution_pie,
    plot_class_distribution_bar,
)
from utils.data_handler import load_text_length
from utils.visualization import (
    plot_text_length_histogram_word,
    plot_text_length_histogram_char,
)
from utils.visualization import print_numeric_summary

eda_app = typer.Typer(help="Exploratory Data Analysis commands")


@eda_app.command()
def distribution(csv_path: str = typer.Option(..., help="Path to CSV file"),
                 label_col: str = typer.Option(..., help="Label column name"),
                 plot_type: str = typer.Option("pie",help="Plot type: pie or bar",),):
    """
    View class distribution as a plot.
    """
    
    class_counts = load_class_distribution(csv_path, label_col)
    print_numeric_summary(list(class_counts.values()),"Class Distribution")

    if plot_type == "pie":
        plot_class_distribution_pie(class_counts)
    elif plot_type == "bar":
        plot_class_distribution_bar(class_counts)
    else:
        raise typer.BadParameter("plot_type must be either 'pie' or 'bar'")
    

@eda_app.command()
def histogram(csv_path: str = typer.Option(..., help="Path to CSV file"),
              text_col: str = typer.Option(..., help="Text column name"),
              plot_type: str = typer.Option("word", help="Histogram type: word or char",),
              bins: str = typer.Option("auto", help="Bins strategy: auto, or any number",),):
    """
    View text length histogram.
    """

    text_counts_word, text_counts_char = load_text_length(
        csv_path, text_col
    )

    if plot_type == "word":
        plot_text_length_histogram_word(text_counts_word, bins=bins)
        print_numeric_summary(text_counts_word, "Word Length")

    elif plot_type == "char":
        plot_text_length_histogram_char(text_counts_char, bins=bins)
        print_numeric_summary(text_counts_word, "Char Length")

    else:
        raise typer.BadParameter("plot_type must be either 'word' or 'char'")


