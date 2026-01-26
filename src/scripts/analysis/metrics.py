import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV (replace with your actual path)
def create_linegraph(metrics_csv: str, output_png: str, title: str):
    df = pd.read_csv(metrics_csv)

    df_epoch = df.groupby("epoch").last().reset_index()

    metrics = {
        "val_auroc": "Validation AUROC",
        "val_map": "Validation MAP (Macro)",
        "val_map_micro": "Validation MAP (Micro)",
        "val_f1": "Validation F1"
    }

    colors = ['orange', 'blue', 'red', 'green']
    i = 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics.items()):
        ax.plot(
            df_epoch["epoch"],
            df_epoch[col],
            marker="o",
            linewidth=2,
            c=colors[i]
        )
        i += 1
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-title', type=str, required=True)

    args = parser.parse_args()

    create_linegraph(args.i, args.o, args.title)