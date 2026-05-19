"""
Analysis script for surrogate classifier.
Generates: loss curve, ROC curve (with AUC), class probability histograms.

Usage:
    python analyze_classifier.py \
        --data_folder /path/to/h5files \
        --weights_directory ../weights \
        --training_config config_surrogate.json \
        --nmax 100000 \
        --output_dir ./plots
"""

import numpy as np
import argparse
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import tensorflow.keras.backend as K

from dataloader import Dataset
from surrogate_model import SurrogateModel
from architecture import Classifier
import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_history(weights_directory: str, model_name: str) -> dict | None:
    pkl_path = os.path.join(weights_directory, model_name, f"{model_name}.pkl")
    # Fallback: sometimes saved next to the checkpoint folder
    alt_path = os.path.join(weights_directory, f"{model_name}.pkl")
    for path in (pkl_path, alt_path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    print(f"[WARNING] Could not find history pickle at {pkl_path} or {alt_path}")
    return None


def build_and_load_model(opt: dict, weights_directory: str) -> tf.keras.Model:
    """Reconstruct the Classifier and load saved weights."""
    model = Classifier(
        opt["NFEAT"],
        opt["NEVT"],
        num_heads=opt["NHEADS"],
        num_transformer=opt["NTRANSF"],
        projection_dim=opt["NDIM"],
    )
    model_name = opt["MODEL_NAME"]
    checkpoint_path = os.path.join(weights_directory, model_name, "checkpoint")
    model.load_weights(checkpoint_path)
    print(f"Loaded weights from {checkpoint_path}")
    return model


def get_predictions(model, dataset: Dataset, batch_size: int) -> np.ndarray:
    """Return raw logits (model output before sigmoid) for all events."""
    inputs = {
        "inputs_particle": dataset.gen[0],
        "inputs_event":    dataset.gen[1],
        "inputs_mask":     dataset.gen[2],
    }
    logits = model.predict(inputs, batch_size=batch_size, verbose=1)[0]  # shape (N, 1)
    return logits[:, 0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(history: dict, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"],     label="Train loss", lw=2)
    ax.plot(epochs, history["val_loss"], label="Val loss",   lw=2, ls="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "loss_curve.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    output_dir: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "roc_curve.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_probabilities(
    probs_data: np.ndarray,
    probs_ref:  np.ndarray,
    weights_data: np.ndarray,
    weights_ref:  np.ndarray,
    output_dir: str,
    n_bins: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 1, n_bins + 1)

    ax.hist(
        probs_data,
        bins=bins,
        weights=weights_data / weights_data.sum(),
        histtype="stepfilled",
        alpha=0.5,
        label="Data  (label = 0)",
        color="steelblue",
    )
    ax.hist(
        probs_ref,
        bins=bins,
        weights=weights_ref / weights_ref.sum(),
        histtype="stepfilled",
        alpha=0.5,
        label="Reference (label = 1)",
        color="tomato",
    )

    ax.set_xlabel("Classifier output probability")
    ax.set_ylabel("Normalised counts")
    ax.set_title("Class Probability Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "class_probabilities.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classifier analysis plots")
    parser.add_argument("--data_folder",      default="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/")
    parser.add_argument("--training_config",  default="config_surrogate.json")
    parser.add_argument("--weights_directory",default="../weights")
    parser.add_argument("--output_dir",       default="./plots")
    parser.add_argument("--nmax",             default=None, type=int)
    parser.add_argument("--batch_size",       default=512, type=int)
    flags = parser.parse_args()

    os.makedirs(flags.output_dir, exist_ok=True)

    # ---- config ----
    opt = utils.LoadJson(flags.training_config)
    model_name = opt["MODEL_NAME"]

    # ---- datasets ----
    reference_files = ["pythia_H1_alphaS0.1180_eplus_5Mevents_prep.h5"]
    data_files      = ["pythia_H1_alphaS0.1500_eplus_5Mevents_prep.h5"]

    data = Dataset(
        data_files,
        flags.data_folder,
        nmax=flags.nmax,
    )
    reference = Dataset(
        reference_files,
        flags.data_folder,
        nmax=flags.nmax,
        norm=data.nmax,
    )

    # ---- model ----
    K.clear_session()
    model = build_and_load_model(opt, flags.weights_directory)

    # ---- predictions ----
    print("Running inference on data...")
    logits_data = get_predictions(model, data, flags.batch_size)

    print("Running inference on reference...")
    logits_ref  = get_predictions(model, reference, flags.batch_size)

    # Sigmoid → class probabilities  (P(label=1 | x))
    probs_data = expit(logits_data)
    probs_ref  = expit(logits_ref)

    # ---- combined arrays for ROC ----
    all_probs   = np.concatenate([probs_data, probs_ref])
    # data → label 0,  reference → label 1  (matches PrepareInputs convention)
    all_labels  = np.concatenate([np.zeros(len(probs_data)), np.ones(len(probs_ref))])
    all_weights = np.concatenate([data.weight, reference.weight])

    # ---- loss curve ----
    history = load_history(flags.weights_directory, model_name)
    if history is not None:
        plot_loss_curve(history, flags.output_dir)
    else:
        print("Skipping loss curve (no history found).")

    # ---- ROC curve ----
    plot_roc_curve(all_labels, all_probs, all_weights, flags.output_dir)

    # ---- class probability histograms ----
    plot_class_probabilities(
        probs_data, probs_ref,
        data.weight, reference.weight,
        flags.output_dir,
    )

    print("\nDone. Plots saved to", flags.output_dir)


if __name__ == "__main__":
    main()