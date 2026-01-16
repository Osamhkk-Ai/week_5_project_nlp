import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


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


def print_numeric_summary(values, title: str):
    arr = np.array(values)

    print(f"\n📊 {title} Summary")
    print("-" * 35)
    print(f"Count   : {len(arr)}")
    print(f"Mean    : {arr.mean():.2f}")
    print(f"Median  : {np.median(arr):.2f}")
    print(f"Std Dev : {arr.std():.2f}")



def plot_text_length_histogram_word(text_counts_word ,bins ,save_path: str = "outputs/visualizations/text_length_histogram_word.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if bins.isdigit():
        bins = int(bins)

    plt.figure()
    plt.hist(text_counts_word, bins = bins)
    plt.title("Text Length Distribution (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()   


def plot_text_length_histogram_char(text_counts_char , bins , save_path: str = "outputs/visualizations/text_length_histogram_char.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if bins.isdigit():
        bins = int(bins)

    plt.figure()
    plt.hist(text_counts_char, bins = bins)
    plt.title("Text Length Distribution (Characters)")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()
