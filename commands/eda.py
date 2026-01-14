import typer
from utils.data_handler import load_class_distribution
from utils.visualization import (
    plot_class_distribution_pie,
    plot_class_distribution_bar,
)

eda_app = typer.Typer(help="Exploratory Data Analysis commands")


@eda_app.command()
def distribution(
    csv_path: str = typer.Option(..., help="Path to CSV file"),
    label_col: str = typer.Option(..., help="Label column name"),
    plot_type: str = typer.Option(
        "pie",
        help="Plot type: pie or bar",
    ),
):
    """
    View class distribution as a plot.
    """

    class_counts = load_class_distribution(csv_path, label_col)

    if plot_type == "pie":
        plot_class_distribution_pie(class_counts)
    elif plot_type == "bar":
        plot_class_distribution_bar(class_counts)
    else:
        raise typer.BadParameter(
            "plot_type must be either 'pie' or 'bar'"
        )
