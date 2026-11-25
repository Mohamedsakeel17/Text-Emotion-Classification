"""Matplotlib visualisation of emotion scores."""
import matplotlib
import matplotlib.pyplot as plt


def plot_emotions(scores: dict, text: str = "", save_path: str = None, show: bool = True):
    """Horizontal bar chart of emotion probabilities (highest at the top)."""
    labels = list(scores.keys())[::-1]
    values = [scores[l] for l in labels]
    top = max(scores, key=scores.get)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1f4e79" if l == top else "#9db7cf" for l in labels]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    title = f'Predicted emotion: {top} ({scores[top]:.0%})'
    ax.set_title(title, fontsize=12, fontweight="bold")
    if text:
        short = text if len(text) <= 70 else text[:67] + "..."
        fig.text(0.5, 0.01, f'"{short}"', ha="center", fontsize=8, style="italic")
    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
