"""
Read jet observables from pythia_H1_jets.root and plot histograms.
"""

import argparse
import os

import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.set_style(hep.style.CMS)

# Mapping from ROOT branch name -> plot key
# Only includes observables that exist in the file
branch_to_key = {
    "jet_pt": "jet_pt",
    "jet_tau10": "jet_tau10",
    "zjet": "zjet",
    "deltaphi" : "deltaphi",
}

dedicated_binning = {
    "jet_pt": np.logspace(np.log10(10), np.log10(100), 7),
    "jet_tau10": np.array(
        [-4.00, -3.15, -2.59, -2.18, -1.86, -1.58, -1.29, -1.05, -0.81, -0.61, 0.00]
    ),
    "zjet": np.linspace(0.2, 1, 11),
    "deltaphi": np.linspace(0, 1, 8),
}

observable_names = {
    "jet_pt": r"$p_{T}^{jet}$ [GeV]",
    "jet_tau10": r"$\mathrm{ln}(\lambda_1^1)$",
    "zjet": r"$z^{jet}$",
    "deltaphi": r"$\Delta\phi^{jet}$ [rad]",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot jet observables from Pythia ROOT files.")
    parser.add_argument("-i", "--input", default="pythia_H1_alphaS118jets.root",
                        help="Input ROOT file (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", default="./alphaS118_plots",
                        help="Output directory for plots (default: %(default)s)")
    parser.add_argument("--alpha-s", default=None, type=float,
                        help="alpha_S value for plot title (auto-detected from filename if not given)")
    return parser.parse_args()


def extract_alphas_from_filename(filename):
    """Try to extract alphaS value from filename like 'pythia_H1_alphaS1136jets.root' -> 0.1136."""
    base = os.path.basename(filename)
    import re
    m = re.search(r"alphaS(\d+)", base)
    if m:
        digits = m.group(1)
        return float(digits) / 10 ** len(digits)
    return None


def main():
    args = parse_args()

    alpha_s = args.alpha_s
    if alpha_s is None:
        alpha_s = extract_alphas_from_filename(args.input)

    title = rf"$\alpha_S = {alpha_s}$" if alpha_s is not None else None

    f = uproot.open(args.input)
    jets = f["jets"]

    os.makedirs(args.output_dir, exist_ok=True)

    for branch, key in branch_to_key.items():
        data = jets[branch].array(library="np")
        # Drop NaNs
        data = data[~np.isnan(data)]

        bins = dedicated_binning[key]
        xlabel = observable_names[key]

        fig, ax = plt.subplots()
        ax.hist(data, bins=bins, density=True, histtype="step", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"$1/\\sigma ~d\\sigma$/d{observable_names[key]}")
        if title:
            ax.set_title(title)
        if key == "jet_pt" or key == "delta_phi":
            ax.set_xscale("log")
        if key != "jet_tau10":
            ax.set_yscale('log')

        outpath = f"{args.output_dir}/{key}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
