from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

LABEL_COLUMNS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

DATA_PATH = Path("data/chexpert/CheXpert-v1.0-small/train.csv")
OUTPUT_PATH = Path("class_distribution.png")


def main() -> None:
    data = pd.read_csv(DATA_PATH, usecols=LABEL_COLUMNS)
    positive_counts = (data == 1).sum().sort_values(ascending=False)
    
    df_plot = positive_counts.reset_index()
    df_plot.columns = ['Label', 'Count']

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))

    _ = sns.barplot(
        data=df_plot,
        x='Count',
        y='Label',
        palette="flare",
        hue='Count',
        legend=False,
        ax=ax
    )

    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + (df_plot['Count'].max() * 0.01),
            p.get_y() + p.get_height() / 2,
            f'{int(width):,}', 
            va='center', 
            fontsize=10, 
            fontweight='bold',
            color="#302C2C"
        )

    ax.set_title("Distribution of Labels in CheXpert", fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("Number of Cases", fontsize=12)
    ax.set_ylabel("")
    
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Plot saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
