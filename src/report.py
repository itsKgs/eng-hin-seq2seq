"""Build the full results report: plots + RESULTS.md.

    python -m src.report

Needs the trained checkpoints (checkpoints/{rnn,gru,lstm}_best.pt and the
*_history.json files written by src.train). Writes:
    results/*.png   (committed to GitHub)
    RESULTS.md      (tables + plots + examples)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")                 # no display needed (Colab / scripts)
import matplotlib.pyplot as plt

from src.config import Config
from src.data import clean_english, clean_hindi, download_dataset, load_pairs
from src.dataset import split_pairs
from src.evaluate import bleu, evaluate_model
from src.train import CHECKPOINT_DIR
from src.utils import get_device

CELLS = ["rnn", "gru", "lstm"]
COLORS = {"rnn": "#d62728", "gru": "#1f77b4", "lstm": "#2ca02c"}
OUT_DIR = "results"
REPORT_PATH = "RESULTS.md"

# Source-length buckets (English tokens) for the BLEU-vs-length plot
LENGTH_BUCKETS = [(1, 3), (4, 5), (6, 7), (8, 9), (10, 100)]
N_EXAMPLES = 15


# ---------------------------------------------------------------- helpers
def load_history(cell):
    path = os.path.join(CHECKPOINT_DIR, f"{cell}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi < 100 else f"{lo}+"


def has_repetition(sentence):
    """True if the same word appears twice in a row (e.g. 'है है')."""
    words = sentence.split()
    return any(a == b for a, b in zip(words, words[1:]))


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)
    return path


# ---------------------------------------------------------------- analysis
def analyse(result, test_pairs, refs, src_lens):
    """Extra per-model statistics computed from the greedy translations."""
    hyps = result["hyps"]
    n = len(hyps)

    # BLEU per source-length bucket
    bucket_bleu, bucket_n = [], []
    for lo, hi in LENGTH_BUCKETS:
        idx = [i for i, L in enumerate(src_lens) if lo <= L <= hi]
        bucket_n.append(len(idx))
        bucket_bleu.append(bleu([hyps[i] for i in idx], [refs[i] for i in idx]))

    # Error statistics
    unk_rate = 100 * sum("<UNK>" in h for h in hyps) / n
    rep_rate = 100 * sum(has_repetition(h) for h in hyps) / n
    hyp_len = sum(len(h.split()) for h in hyps)
    ref_len = sum(len(r.split()) for r in refs)
    exact = 100 * sum(h == r for h, r in zip(hyps, refs)) / n
    ref_rep = 100 * sum(has_repetition(r) for r in refs) / n

    result.update({
        "bucket_bleu": bucket_bleu,
        "bucket_n": bucket_n,
        "unk_rate": unk_rate,
        "rep_rate": rep_rate,
        "length_ratio": hyp_len / ref_len,
        "exact_match": exact,
        "ref_rep_rate": ref_rep,
    })
    return result


# ---------------------------------------------------------------- plots
def plot_loss_curves(histories):
    """One panel per model: train vs val loss, best epoch marked."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4),
                             sharey=True)
    if len(histories) == 1:
        axes = [axes]
    for ax, (cell, hist) in zip(axes, histories.items()):
        epochs = [h["epoch"] for h in hist["history"]]
        train = [h["train_loss"] for h in hist["history"]]
        val = [h["val_loss"] for h in hist["history"]]
        best = min(hist["history"], key=lambda h: h["val_loss"])

        ax.plot(epochs, train, "-o", ms=3, color=COLORS[cell], alpha=0.5, label="train")
        ax.plot(epochs, val, "-o", ms=3, color=COLORS[cell], label="val")
        ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
        ax.annotate(f"best val {best['val_loss']:.3f}\n(epoch {best['epoch']})",
                    xy=(best["epoch"], best["val_loss"]),
                    xytext=(10, 25), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_title(f"{cell.upper()}  ({hist['parameters']:,} params)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("cross-entropy loss per token")
    fig.suptitle("Train vs validation loss (train: teacher forcing 0.5, val: 1.0)",
                 fontsize=10)
    return save(fig, "loss_curves.png")


def plot_val_comparison(histories):
    """All validation curves on one plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for cell, hist in histories.items():
        epochs = [h["epoch"] for h in hist["history"]]
        val = [h["val_loss"] for h in hist["history"]]
        best = min(hist["history"], key=lambda h: h["val_loss"])
        ax.plot(epochs, val, "-o", ms=3, color=COLORS[cell], label=cell.upper())
        ax.plot(best["epoch"], best["val_loss"], "*", ms=14, color=COLORS[cell])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss: RNN vs GRU vs LSTM (★ = saved checkpoint)")
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, "val_loss_comparison.png")


def plot_train_comparison(histories):
    """All training curves on one plot (shows how fast each model memorises)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for cell, hist in histories.items():
        epochs = [h["epoch"] for h in hist["history"]]
        train = [h["train_loss"] for h in hist["history"]]
        ax.plot(epochs, train, "-o", ms=3, color=COLORS[cell], label=cell.upper())
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Training loss: how fast each model fits the training set")
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, "train_loss_comparison.png")


def plot_bleu(results):
    """Grouped bars: overall / short / long BLEU."""
    groups = ["Overall", "Short (≤5 words)", "Long (≥10 words)"]
    keys = ["bleu", "bleu_short", "bleu_long"]
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, r in enumerate(results):
        xs = [i + j * width for i in range(len(groups))]
        vals = [r[k] for k in keys]
        bars = ax.bar(xs, vals, width, color=COLORS[r["cell_type"]],
                      label=r["cell_type"].upper())
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks([i + width * (len(results) - 1) / 2 for i in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("BLEU (sacrebleu)")
    ax.set_title("Test BLEU by sentence length")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return save(fig, "bleu_comparison.png")


def plot_bleu_by_length(results):
    """BLEU per source-length bucket: the fixed-size bottleneck in one picture."""
    labels = [bucket_label(lo, hi) for lo, hi in LENGTH_BUCKETS]
    counts = results[0]["bucket_n"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        ax.plot(labels, r["bucket_bleu"], "-o", color=COLORS[r["cell_type"]],
                label=r["cell_type"].upper())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, counts)])
    ax.set_xlabel("English sentence length (tokens)")
    ax.set_ylabel("BLEU")
    ax.set_title("BLEU vs source length (one fixed-size vector for every length)")
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, "bleu_by_length.png")


def plot_summary(results):
    """2x3 panel: size, time, losses, BLEU."""
    panels = [
        ("parameters", "Parameters", ",.0f"),
        ("train_seconds", "Total training time (s)", ".0f"),
        ("best_epoch", "Best epoch (lowest val loss)", ".0f"),
        ("val_loss", "Best validation loss", ".3f"),
        ("test_loss", "Test loss", ".3f"),
        ("bleu", "Test BLEU", ".2f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    names = [r["cell_type"].upper() for r in results]
    colors = [COLORS[r["cell_type"]] for r in results]
    for ax, (key, title, spec) in zip(axes.flat, panels):
        vals = [r[key] if r[key] is not None else 0 for r in results]
        bars = ax.bar(names, vals, color=colors)
        ax.bar_label(bars, labels=[format(v, spec) for v in vals], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.15)
        ax.set_ylim(bottom=0)
    fig.suptitle("Model summary", fontsize=12)
    return save(fig, "model_summary.png")


def plot_errors(results):
    """Error analysis of the greedy translations."""
    panels = [
        ("unk_rate", "% outputs containing <UNK>", ".1f"),
        ("rep_rate", "% outputs with the same word twice in a row", ".1f"),
        ("length_ratio", "Output length / reference length", ".2f"),
        ("exact_match", "% exact-match translations", ".1f"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    names = [r["cell_type"].upper() for r in results]
    colors = [COLORS[r["cell_type"]] for r in results]
    for ax, (key, title, spec) in zip(axes, panels):
        vals = [r[key] for r in results]
        bars = ax.bar(names, vals, color=colors)
        ax.bar_label(bars, labels=[format(v, spec) for v in vals], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.15)
        ax.set_ylim(bottom=0)
        if key == "length_ratio":
            ax.axhline(1.0, color="gray", ls="--", lw=1)
        if key == "rep_rate":
            # Baseline: how often the REFERENCES themselves repeat a word
            ax.axhline(results[0]["ref_rep_rate"], color="gray", ls="--", lw=1,
                       label="references")
            ax.legend(fontsize=7)
    fig.suptitle("Error analysis (greedy decoding on the test set)", fontsize=11)
    return save(fig, "error_analysis.png")


# ---------------------------------------------------------------- report
def write_report(results, test_pairs, refs, histories):
    def row(label, key, spec):
        vals = []
        for r in results:
            v = r.get(key)
            vals.append("-" if v is None else format(v, spec))
        return f"| {label} | " + " | ".join(vals) + " |"

    names = [r["cell_type"].upper() for r in results]
    n_test = len(test_pairs)
    lines = [
        "# Results: English → Hindi Seq2Seq (RNN vs GRU vs LSTM)",
        "",
        "Basic Encoder–Decoder without attention, trained on the Tatoeba / ManyThings "
        "English–Hindi pairs (80/10/10 split, seed 42). Same data, vocabulary, "
        "hyperparameters and seed for all three models; only the recurrent cell changes.",
        f"Test set: {n_test} sentence pairs. Decoding: greedy. BLEU: sacrebleu corpus BLEU "
        "on our word-level tokenization, single reference.",
        "",
        "## Metrics",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        row("Parameters", "parameters", ","),
        row("Total training time (s)", "train_seconds", ".0f"),
        row("Best epoch (lowest val loss)", "best_epoch", "d"),
        row("Train loss at best epoch", "train_loss", ".3f"),
        row("Best validation loss", "val_loss", ".3f"),
        row("Test loss", "test_loss", ".3f"),
        row("Test BLEU", "bleu", ".2f"),
        row(f"BLEU short (≤5 words, n={results[0]['n_short']})", "bleu_short", ".2f"),
        row(f"BLEU long (≥10 words, n={results[0]['n_long']})", "bleu_long", ".2f"),
        row("% outputs with <UNK>", "unk_rate", ".1f"),
        row("% outputs with a word repeated twice in a row", "rep_rate", ".1f"),
        row("  (same statistic for the references)", "ref_rep_rate", ".1f"),
        row("Output / reference length", "length_ratio", ".2f"),
        row("% exact-match translations", "exact_match", ".1f"),
        "",
        "## Loss curves",
        "",
        "![Train vs validation loss](results/loss_curves.png)",
        "",
        "![Validation loss comparison](results/val_loss_comparison.png)",
        "",
        "![Training loss comparison](results/train_loss_comparison.png)",
        "",
        "Note: training loss uses teacher forcing 0.5 and validation loss uses 1.0, so "
        "the two are not directly comparable. Overfitting shows as the validation loss "
        "rising while the training loss keeps falling.",
        "",
        "## BLEU",
        "",
        "![BLEU comparison](results/bleu_comparison.png)",
        "",
        "![BLEU by sentence length](results/bleu_by_length.png)",
        "",
        "## Model summary",
        "",
        "![Model summary](results/model_summary.png)",
        "",
        "## Error analysis",
        "",
        "![Error analysis](results/error_analysis.png)",
        "",
        f"## Qualitative examples (first {N_EXAMPLES} test sentences)",
        "",
    ]
    for i in range(min(N_EXAMPLES, n_test)):
        en, _ = test_pairs[i]
        lines.append(f"**{i + 1}. {en}**")
        lines.append("")
        lines.append(f"- REF: {refs[i]}")
        for r in results:
            lines.append(f"- {r['cell_type'].upper()}: {r['hyps'][i]}")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved", REPORT_PATH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    cfg = Config()

    # Same test split as training / evaluation
    pairs = load_pairs(download_dataset(cfg.data_dir))
    _, _, test_pairs = split_pairs(pairs, seed=cfg.seed)
    refs = [clean_hindi(hi) for _, hi in test_pairs]
    src_lens = [len(clean_english(en).split()) for en, _ in test_pairs]

    histories = {c: h for c in CELLS if (h := load_history(c)) is not None}
    results = []
    for cell in CELLS:
        r = evaluate_model(cell, test_pairs, device)
        if r is not None:
            results.append(analyse(r, test_pairs, refs, src_lens))
            print(f"evaluated {cell}: BLEU {r['bleu']:.2f}")

    if not results:
        print("No checkpoints found. Train first: python -m src.train --cell_type gru")
        return

    plot_loss_curves(histories)
    plot_val_comparison(histories)
    plot_train_comparison(histories)
    plot_bleu(results)
    plot_bleu_by_length(results)
    plot_summary(results)
    plot_errors(results)
    write_report(results, test_pairs, refs, histories)


if __name__ == "__main__":
    main()
