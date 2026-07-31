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
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.tensorflow.keras as hvd

from dataloader import Dataset
from surrogate_model import SurrogateModel
from architecture import Classifier
import utils
import mplhep as hep
hep.style.use("CMS")


# Feature name look-up tables (shared by both plot functions)
EVENT_NAMES = {
    "0": r"$\log(Q^2)$",
    "1": r"$y$",
    "2": r"$e_{pT}/Q$",
    "3": r"$e_{\eta}$",
    "4": r"$e_{\phi}$",
}

PARTICLE_NAMES = {
    "0": r"$\eta_p - \eta_e$",
    "1": r"$\phi_p - \phi_e - \pi$",
    "2": r"$\log(p_T)$",
    "3": r"$\log(p_T/Q)$",
    "4": r"$\log(E/Q)$",
    "5": r"$\log(E)$",
    "6": r"$\sqrt{(\eta_p - \eta_e)^2 + (\phi_p - \phi_e)^2}$",
    "7": "Absolute Charge",
}

# Display labels for parameter-value event features (e.g. alpha_s), keyed by
# the manifest/config parameter name. Falls back to the raw name if missing.
PARAM_LABELS = {
    "alpha_s": r"$\alpha_s$",
}


def _format_param_label(params, param_names):
    """Build a plot-friendly label from a manifest entry's parameter values, e.g. '$\\alpha_s$ = 0.15'."""
    return ", ".join(f"{PARAM_LABELS.get(name, name)} = {params[name]}" for name in param_names)


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


def build_and_load_model(flags, opt) -> tf.keras.Model:
    """Reconstruct the Classifier and load saved weights."""
    surrogate = SurrogateModel(
        config_file=flags.training_config,
        verbose=hvd.rank() == 0,
        weights_directory=flags.weights_directory,
    )
    surrogate.PrepareModel()
    model_name = opt["MODEL_NAME"]
    checkpoint_path = os.path.join(flags.weights_directory, model_name, "checkpoint")
    surrogate.model.load_weights(checkpoint_path).expect_partial()
    print(f"Loaded weights from {checkpoint_path}")
    return surrogate


def evaluate_model(surrogate, dataset: Dataset, batch_size: int) -> np.ndarray:
    """Return per-event model weights for the given dataset."""
    weights = surrogate.reweight(dataset.gen, surrogate.model_ema, batch_size=batch_size)
    return weights


def get_hist_binning(arrays, nbins):
    """Compute shared bin edges from a list of arrays, ignoring non-finite values."""
    clean_arrays = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            clean_arrays.append(arr)

    if not clean_arrays:
        return nbins

    values = np.concatenate(clean_arrays)
    return nbins if values.size == 0 else np.histogram_bin_edges(values, bins=nbins)


def undo_standardizing(dataloader):
    """Invert preprocessing transforms in place."""
    dataloader.part, dataloader.event = dataloader.revert_standardize(
        dataloader.gen[0], dataloader.gen[1], dataloader.gen[-1]
    )
    dataloader.mask = dataloader.gen[-1]
    del dataloader.gen
    gc.collect()


def gather_data(dataloader, event_weights=None):
    n_events = dataloader.event.shape[0]
    event_plot_mask = np.ones(n_events, dtype=bool)
    particle_plot_mask = dataloader.mask

    flat_part = dataloader.part.reshape((-1, dataloader.part.shape[-1]))
    flat_particle_plot_mask = particle_plot_mask.reshape(-1)
    selected_part = flat_part[flat_particle_plot_mask]

    particle_weights = np.broadcast_to(dataloader.weight[:, None], dataloader.mask.shape)
    flat_particle_weights = particle_weights.reshape(-1)
    selected_particle_weight = flat_particle_weights[flat_particle_plot_mask]

    if event_weights is not None:
        particle_event_weights = np.broadcast_to(event_weights[:, None], dataloader.mask.shape)
        flat_particle_event_weights = particle_event_weights.reshape(-1)
        selected_particle_event_weights = flat_particle_event_weights[flat_particle_plot_mask]

        dataloader.particle_event_weight = hvd.allgather(
            tf.constant(selected_particle_event_weights)
        ).numpy()
        dataloader.model_weights = hvd.allgather(
            tf.constant(event_weights)
        ).numpy()

    dataloader.part = hvd.allgather(tf.constant(selected_part)).numpy()
    dataloader.event = hvd.allgather(tf.constant(dataloader.event)).numpy()
    selected_weight = dataloader.weight[event_plot_mask]
    dataloader.weight = hvd.allgather(tf.constant(selected_weight)).numpy()
    dataloader.particle_weight = hvd.allgather(tf.constant(selected_particle_weight)).numpy()


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

    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _density_vals(values: np.ndarray, weights: np.ndarray, bins) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (density, edges) for a weighted histogram, equivalent to density=True
    in plt.hist.  Non-finite values (and their weights) are dropped before binning.
    """
    mask = np.isfinite(values)
    v, w = values[mask], weights[mask]
    counts, edges = np.histogram(v, bins=bins, weights=w)
    widths = np.diff(edges)
    total  = counts.sum()
    dens   = counts / (total * widths) if total > 0 else np.zeros_like(counts, dtype=float)
    return dens, edges


def _save_feature_plot(ax_main, ax_ratio, fig, xlabel, output_path):
    """Apply common formatting to both panels and save."""
    ax_main.set_ylabel("Normalized entries", fontsize=18)
    ax_main.legend(fontsize=14)
    ax_main.grid(alpha=0.3)
    ax_ratio.set_xlabel(xlabel, fontsize=18)
    ax_ratio.set_ylabel("Data/Ref.", fontsize=18)
    ax_ratio.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _three_panel_hists(ax_main, ax_ratio, data_vals, ref_vals, data_w, data_rw, ref_w, binning, labels):
    """
    Top panel  – density histograms for the three distributions:
      1. Unweighted data  – step outline, MC weights only
      2. Reweighted data  – filled + alpha, MC weights × model weights
      3. Reference        – step outline, MC weights only

    Bottom panel – ratio of reweighted data to reference.
    A horizontal guide line is drawn at ratio = 1.

    labels: dict with keys "unweighted", "reweighted", "reference".
    """
    dens_unw, edges = _density_vals(data_vals, data_w,  binning)
    dens_rew, _     = _density_vals(data_vals, data_rw, binning)
    dens_ref, _     = _density_vals(ref_vals,  ref_w,   binning)

    # Grab the default colour cycle so main and ratio panels share colours
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_unw, c_rew, c_ref = prop_cycle[0], prop_cycle[1], prop_cycle[2]

    # ---- main panel ----
    ax_main.stairs(dens_rew, edges, color=c_rew, lw=1.5, fill=True, alpha=0.4,
                   label=labels["reweighted"])
    ax_main.stairs(dens_unw, edges, color=c_unw, lw=1.5,
                   label=labels["unweighted"])
    ax_main.stairs(dens_ref, edges, color=c_ref, lw=1.5,
                   label=labels["reference"])

    # ---- ratio panel ----
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_to_ref = np.where(dens_ref > 0, dens_rew / dens_ref, np.nan)
        unweighted_ratio_to_ref = np.where(dens_ref > 0, dens_unw / dens_ref, np.nan)

    ax_ratio.stairs(ratio_to_ref, edges, color=c_rew, lw=1.5)
    ax_ratio.stairs(unweighted_ratio_to_ref, edges, color=c_unw, lw=1.5)
    ax_ratio.axhline(1.0, color="black", lw=0.8, ls=":")

    # Y-limits: centre on 1, expand symmetrically to cover the actual range
    ax_ratio.set_ylim(0.5, 1.5)


def plot_all_event_features(reweighted, target, output_dir: str, labels: dict, nbins: int = 50) -> None:
    """
    For every event-level feature plot:
      • Unweighted reweighted-sample  (MC weight only)   -- e.g. the reference sample
      • Model-reweighted reweighted-sample  (MC weight × model weight)
      • Target  (MC weight only)  -- the fixed sample the reweighting is trying to match, e.g. data

    `reweighted` must have `.model_weights` attached (see `gather_data`); `target` does not need it.

    labels: dict with keys "unweighted", "reweighted", "reference".
    """
    if hvd.rank() != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    n_features = reweighted.event.shape[-1]

    for feature in range(n_features):
        reweighted_vals = reweighted.event[:, feature]
        target_vals     = target.event[:, feature]

        binning = get_hist_binning(
            [reweighted_vals[np.isfinite(reweighted_vals)], target_vals[np.isfinite(target_vals)]],
            nbins,
        )

        # Composite weights: MC weight alone vs MC weight × model weight
        reweighted_w  = reweighted.weight
        reweighted_rw = reweighted.weight * reweighted.model_weights
        target_w      = target.weight

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        )
        _three_panel_hists(
            ax_main, ax_ratio, reweighted_vals, target_vals,
            reweighted_w, reweighted_rw, target_w, binning, labels,
        )

        xlabel = EVENT_NAMES.get(str(feature), f"Event feature {feature}")
        output_path = os.path.join(output_dir, f"event_feature_{feature}.png")
        _save_feature_plot(ax_main, ax_ratio, fig, xlabel, output_path)


def plot_all_particle_features(reweighted, target, output_dir: str, labels: dict, nbins: int = 50) -> None:
    """
    For every particle-level feature plot:
      • Unweighted reweighted-sample  (particle MC weight only)  -- e.g. the reference sample
      • Model-reweighted reweighted-sample  (particle MC weight × per-event model weight)
      • Target  (particle MC weight only)  -- the fixed sample the reweighting is trying to match, e.g. data

    `reweighted` must have `.particle_event_weight` attached (see `gather_data`); `target` does not need it.

    labels: dict with keys "unweighted", "reweighted", "reference".
    """
    if hvd.rank() != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    n_features = reweighted.part.shape[-1]

    for feature in range(n_features):
        reweighted_vals = reweighted.part[:, feature]
        target_vals     = target.part[:, feature]

        binning = get_hist_binning(
            [reweighted_vals[np.isfinite(reweighted_vals)], target_vals[np.isfinite(target_vals)]],
            nbins,
        )

        # Composite weights: particle MC weight alone vs × per-event model weight
        reweighted_w  = reweighted.particle_weight
        reweighted_rw = reweighted.particle_weight * reweighted.particle_event_weight
        target_w      = target.particle_weight

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        )
        _three_panel_hists(
            ax_main, ax_ratio, reweighted_vals, target_vals,
            reweighted_w, reweighted_rw, target_w, binning, labels,
        )

        xlabel = PARTICLE_NAMES.get(str(feature), f"Particle feature {feature}")
        output_path = os.path.join(output_dir, f"particle_feature_{feature}.png")
        _save_feature_plot(ax_main, ax_ratio, fig, xlabel, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classifier analysis plots")
    parser.add_argument("--data_folder",       default="../pythia_h5/")
    parser.add_argument("--training_config",   default="config_surrogate.json")
    parser.add_argument("--weights_directory", default="/projects/bhvk/rmilton/H1Unfold_April2026_training/weights/")
    parser.add_argument("--output_dir",        default="./plots")
    parser.add_argument("--nmax",              default=None, type=int)
    parser.add_argument("--batch_size",        default=512, type=int)
    parser.add_argument(
        "--data_file",
        default="pythia_H1_alphaS0.1500_eplus_1Mevents_prep.h5",
        help="Non-reference (target) manifest file the reference sample is reweighted toward",
    )
    flags = parser.parse_args()

    os.makedirs(flags.output_dir, exist_ok=True)

    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # ---- config ----
    opt = utils.LoadJson(flags.training_config)
    model_name = opt["MODEL_NAME"]
    param_names = opt["PARAMETERS"]
    manifest = utils.LoadManifest("dataset_manifest.yaml")

    # Label the parameter-value event feature columns (appended after the base
    # NEVT physics columns) for the per-feature plots below.
    for i, name in enumerate(param_names):
        EVENT_NAMES[str(opt["NEVT"] + i)] = PARAM_LABELS.get(name, name)

    # ---- datasets ----
    reference_entry = utils.GetReferenceFile(manifest)
    data_entry = utils.GetManifestEntry(manifest, flags.data_file)

    datasets = {
        "data": {
            "files": [data_entry["path"]],
            "label": _format_param_label(data_entry["parameters"], param_names) + " (target)",
        },
        "reference": {
            "files": [reference_entry["path"]],
            # The reference is physically the alpha_s=0.118 sample, so label the plots
            # with its own parameters, but feed the network the *target's* parameter
            # values -- that is the point at which we want the likelihood ratio, and it
            # matches how the reference was labelled during training.
            "params": data_entry["parameters"],
            "label":     _format_param_label(reference_entry["parameters"], param_names) + " (ref.)",
            "label_rew": "Reweighted " + _format_param_label(reference_entry["parameters"], param_names) + " (ref.)",
        },
    }

    labels = {
        "unweighted": datasets["reference"]["label"],
        "reweighted": datasets["reference"]["label_rew"],
        "reference":  datasets["data"]["label"],
    }

    print("Loading data")
    data = Dataset(
        datasets["data"]["files"],
        flags.data_folder,
        nmax=flags.nmax,
        rank=hvd.rank(),
        size=hvd.size(),
        param_names=param_names,
        file_params=utils.GetFileParams(manifest, datasets["data"]["files"], param_names),
    )
    print("Loading reference")
    reference = Dataset(
        datasets["reference"]["files"],
        flags.data_folder,
        nmax=flags.nmax,
        norm=data.nmax,
        rank=hvd.rank(),
        size=hvd.size(),
        param_names=param_names,
        file_params=[datasets["reference"]["params"]],
    )
    print("Done loading")

    # ---- model ----
    K.clear_session()
    model = build_and_load_model(flags, opt)

    # ---- predictions ----
    print("Evaluating model weights")
    weights_reference = evaluate_model(model, reference, flags.batch_size)

    # ---- pre-process arrays ----
    undo_standardizing(data)
    undo_standardizing(reference)

    # gather_data attaches .model_weights / .particle_event_weight to `reference`
    gather_data(reference, weights_reference)
    gather_data(data)

    # ---- loss curve ----
    history = load_history(flags.weights_directory, model_name)
    if history is not None:
        plot_loss_curve(history, flags.output_dir)
    else:
        print("Skipping loss curve (no history found).")

    # ---- feature distributions ----
    # `reference` is the sample being reweighted to match the fixed `data` target.
    print("Plotting event features")
    plot_all_event_features(reweighted=reference, target=data, output_dir=flags.output_dir, labels=labels)

    print("Plotting particle features")
    plot_all_particle_features(reweighted=reference, target=data, output_dir=flags.output_dir, labels=labels)

    print("\nDone. Plots saved to", flags.output_dir)


if __name__ == "__main__":
    main()