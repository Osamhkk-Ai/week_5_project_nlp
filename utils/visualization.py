import matplotlib.pyplot as plt
from pathlib import Path


def plot_class_distribution_pie(
    class_counts: dict,
    save_path: str = "outputs/visualizations/class_distribution_pie.png",
):
    """
    Plot and save a pie chart for class distribution.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Class Distribution")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_distribution_bar(
    class_counts: dict,
    save_path: str = "outputs/visualizations/class_distribution_bar.png",
):
    """
    Plot and save a bar chart for class distribution.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
