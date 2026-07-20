# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Simulation-based inference (SBI) study for extracting alpha_s from H1 (HERA ep DIS)-like Pythia
simulations. The pipeline: generate Pythia events at a given alpha_s → cluster final-state particles
into jets → convert/preprocess events into ML-ready tensors → train a transformer classifier to
distinguish a "data" Pythia sample from a fixed "reference" Pythia sample → use the classifier's
learned likelihood ratio to reweight the data sample toward the reference → make QA/validation plots
of the reweighted distributions. This is a classifier-reweighting approach to unbinned inference, not
a traditional binned unfolding.

`H1Unfold/` is a separate, unrelated project living in this same directory tree — ignore it.

## Pipeline stages and where they live

1. **Event generation** (`pythia_generation/pythia_H1.cc`) — C++ Pythia8 DIS generator (positron/electron
   on proton, 600 GeV cms-ish HERA kinematics, `Q2min=150`, `0.2<y<0.7`). Takes `alpha_s`, `nEvents`,
   `lepton_id` (11 or -11) as positional CLI args. Sets `alpha_s` identically for hard process, ISR and
   FSR. Writes a ROOT file with three TTrees: `events` (Q2, W, x, y, weight), `electron` (scattered
   lepton kinematics), `particles` (all other visible final-state particles, neutrinos excluded, with
   `eta` and `pT` fiducial cuts already applied). Output path is hardcoded to
   `.../unbinned_inference/pythia_files/pythia_H1_600GeV_alphaS<value>_<eplus|eminus>_<N>events.root`.
   No build script is checked into this repo — the compiled `pythia_generation/pythia_H1` binary
   (gitignored) is built externally against Pythia8/ROOT/LHAPDF, same as PYTHIA's `main341.cc` example.

2. **Jet clustering** (`pythia_generation/cluster_jets.py`) — reads the generation ROOT file with uproot,
   reconstructs the virtual photon `q` via the sigma method, clusters HFS particles with FastJet
   `kt_algorithm` (R=1.0) in the lab frame and optionally the Breit frame (`--use_breit`), and computes
   substructure observables (`tau_10`, `zjet`, `deltaphi`). `--use_centauro` additionally boosts to the
   Breit frame, writes particles to a temp ROOT file, and shells out to
   `pythia_generation/run_centauro.sh`, which compiles `cluster_centauro.cxx` (FastJet Centauro plugin,
   from the CVMFS LCG_109 view) on the fly and runs it as a subprocess. `cluster_jets_iterations.sh`
   batches a large input file into 10 chunks (`--start_event`/`--end_event`), runs clustering on each,
   and merges the outputs with `hadd`. `pythia_generation/plot_jets.py` plots the resulting jet
   observable histograms from a clustered ROOT file. **This clustered-jets path is a separate
   substructure-analysis track — the ML classifier below does not consume `clustered_pythia_files/`,
   it consumes particle-level features produced independently by `prepare_pythia_data.py`.**

3. **ROOT → HDF5 conversion** (`scripts/prepare_pythia_data.py`) — reads one or more generation ROOT
   files matching an `--input_string` glob, reapplies the fiducial cuts, computes `pt/phi/eta` for
   HFS particles, sorts by `pt` descending, pads/truncates to `max_nonzero_particles` (132), and writes
   `gen_particle_features` (pt, eta, phi) and `gen_event_features` (Q2, y, e_px, e_py, e_pz, weight) to
   an `.h5` file.

4. **Preprocessing** (`scripts/preprocess_pythia.py`) — loads a `.h5` file and derives the actual
   network input features: particle-level `[delta_eta, delta_phi, log_pt, log_pt_rel, log_e_rel, log_e,
   delta_r]` (relative to the scattered electron and to Q) and event-level `[log_Q2, y, e_pt/Q, e_eta,
   e_phi, weight]`. Writes a new `*_prep.h5` file — this is the file consumed by training/analysis.

5. **End-to-end generation/prep wrapper** (`scripts/generate_and_preprocess_pythia.sh`) — takes
   `<alpha_s> <n_events> <lepton_id>`, runs the pythia_H1 binary (currently commented out — assumes the
   ROOT file already exists), then `prepare_pythia_data.py`, then `preprocess_pythia.py` in sequence.
   `scripts/run_all_alphaS.sh` loops this over `alpha_s = 0.10 … 0.20` at 1M events each.

6. **Classifier / reweighting model** (`scripts/`):
   - `dataloader.py` — `Dataset` class: loads `*_prep.h5`, splits particle weight column out of event
     features, **standardizes with hardcoded mean/std constants** (computed from a past reference run,
     not recomputed per-dataset — see commented-out code for how they were derived), and normalizes
     sample weights so two datasets of different size/statistics are comparable
     (`weight *= norm/nmax`). Supports MPI-style sharding via `rank`/`size` (Horovod ranks).
   - `architecture.py` — `Classifier`: a PET-style (Point-Edge Transformer) point-cloud model. Body =
     local kNN feature extraction + multi-head-attention transformer blocks with `LayerScale`; head =
     class-token cross-attention conditioned on event-level features (FiLM-style scale/shift). Outputs
     both a classification logit and an auxiliary event-feature reconstruction (trained with weighted
     BCE + 0.1×MSE). Trains with **two separate optimizers** (body vs head, both `Lion` wrapped in
     `hvd.DistributedOptimizer`) via a custom `train_step`/`GradientTape`, not vanilla `model.fit`
     gradients. Maintains an EMA copy of the weights (`model_ema`) — **inference/reweighting should use
     `model_ema`, not the raw trained model** (see `analyze_classifier.py`).
   - `layers.py` — attention/layer-scale/stochastic-depth building blocks used by `architecture.py`.
   - `surrogate_model.py` — `SurrogateModel` orchestrates config loading, dataset assignment
     (`self.data` = sample being reweighted, `self.reference` = fixed target), model compilation with
     cosine-decay LR schedules (warmup + Horovod-size-scaled peak LR), training loop with Horovod
     callbacks, checkpointing, and `reweight()` which converts classifier logits to per-event weights
     via `f/(1-f)` where `f = sigmoid(logit)` (standard likelihood-ratio/classifier-reweighting trick).
   - `train.py` — entry point. **Currently hardcodes** `reference_files = alphaS0.1180` and
     `data_files = alphaS0.1500` (both 5M-event samples) — change these to retrain on a different pair.
     Must be launched under Horovod (`hvd.init()` is called unconditionally; run via `horovodrun`/`mpirun`).
   - `utils.py` — `LoadJson(file_name, base_path="../configs")`: despite the name and `.json` extension,
     config files are parsed with `yaml.safe_load`, not `json.load` — `configs/config_surrogate.json`
     actually uses YAML flow-mapping syntax (single-quoted keys, trailing comma), which is not valid
     JSON but is valid YAML. Because `base_path` defaults to `../configs`, training/analysis scripts
     assume they are launched **from inside `scripts/`**.

7. **Analysis / QA plotting**:
   - `scripts/analyze_classifier.py` — reconstructs the `Classifier`, loads checkpoint weights,
     evaluates `model_ema` on the "data" sample to get reweighting factors, reverts standardization, and
     produces the training loss curve plus per-feature (event- and particle-level) three-panel plots:
     unweighted data vs. model-reweighted data vs. reference, with a ratio-to-reference panel.
   - `scripts/QA_plots.py` — pre-training sanity-check plots comparing raw (unweighted) particle/event
     feature distributions across multiple Pythia alpha_s samples (and references some H1 real-data/
     Rapgap/Djangoh h5 paths under `/global/cfs/cdirs/m3246/H1/...` for future comparisons). Also
     defines an `hvd.init()` at import time.
   - `plot_utils.py` (repo root) — a large (~2600 line) legacy plotting module referencing
     `Rapgap`/`Djangoh`/`options.name_translate`/unfolded weights — this is leftover from a different
     (unfolding-style) analysis and **is not imported by any current script in this repo**. Treat as
     dead/reference code, not part of the active pipeline.

## Data flow / directory layout

```
pythia_files/            raw Pythia ROOT output (pythia_H1.cc)              [gitignored]
clustered_pythia_files/  FastJet-clustered ROOT files (cluster_jets.py)     [gitignored, separate track]
pythia_h5/                *.h5          -> prepare_pythia_data.py output
                           *_prep.h5     -> preprocess_pythia.py output (network-ready)
weights/                 model checkpoints per MODEL_NAME + training history .pkl  [gitignored]
plots/, scripts/plots*/  QA and analysis output figures                    [gitignored]
```

`pythia_files/`, `pythia_h5/`, `clustered_pythia_files/`, `weights/`, `plots/`, compiled binaries
(`pythia_generation/pythia_H1`, `pythia_generation/cluster_centauro`), and `*.txt` logs are all
gitignored — this repo tracks code only, all data/model artifacts live on CFS and are regenerated by
the pipeline scripts above.

## Running the pipeline

All commands assume NERSC/Perlmutter with CVMFS available (`run_centauro.sh` sources the LCG_109
x86_64-el9-gcc15-opt view for `fastjet-config`/Centauro) and a Python env with `uproot`, `awkward`,
`fastjet`, `h5py`, `tensorflow`, `horovod`, `mplhep`, `scikit-learn` (no `requirements.txt`/env file is
checked into the repo).

```bash
# Full generation -> h5 prep pipeline for one alpha_s point (run from scripts/)
./generate_and_preprocess_pythia.sh <alpha_s> <n_events> <lepton_id>   # e.g. 0.118 5000000 -11

# Sweep alpha_s = 0.10..0.20 at 1M events each
./run_all_alphaS.sh

# Jet clustering on an existing generation ROOT file
python pythia_generation/cluster_jets.py --input <in.root> --output <out.root> \
    --use_breit --use_centauro --lepton_beam -11

# Batched clustering + hadd merge over a big file (edit paths at top of script first)
bash pythia_generation/cluster_jets_iterations.sh

# Train the classifier (from scripts/, under Horovod)
horovodrun -np <N> python train.py --training_config config_surrogate.json

# Post-training analysis plots
python analyze_classifier.py --data_folder <h5 dir> --weights_directory ../weights \
    --training_config config_surrogate.json --output_dir ./plots

# Pre-training QA comparison plots across raw samples
python QA_plots.py --mc_keys pythia_alphaS15 pythia_alphaS118
```

There is no test suite, linter, or CI configuration in this repo.
