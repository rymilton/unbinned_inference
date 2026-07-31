# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Neural simulation-based inference (NSBI) to determine the **Pythia generation parameters** that best
reproduce **unbinned unfolded H1 data** (HERA ep DIS). There is more than one parameter — `alpha_s` is
simply the first one implemented — so **do not write alpha_s-specific code**; the manifest and config
plumbing exist precisely so parameters can be added without touching the model or dataloader.

The pipeline: generate Pythia events at a given set of generation parameters → convert/preprocess
events into ML-ready tensors → train a transformer classifier to distinguish a "data" Pythia sample
from a fixed "reference" Pythia sample → the classifier's learned likelihood ratio (`f/(1-f)`) is the
**surrogate likelihood**, which is ultimately evaluated against unfolded H1 data to infer the
parameters → QA/validation plots of the reweighted distributions. This is a classifier-reweighting
approach to unbinned inference, not a traditional binned unfolding.

The classifier is **parameterized**: each sample's generation-parameter values are fed to the network
as extra event-level input features, so one model covers the parameter space continuously rather than
one model per point. See "Parameterized classifier" below — the reference-replication step is
essential and non-obvious.

**Current status:** development/validation runs are Pythia-against-Pythia (closure tests, where the
"data" side is simulation at a known parameter point). Unfolded H1 data is the eventual target — the
H1/Rapgap/Djangoh paths referenced in `QA_plots.py` point that way.

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
   - `dataloader.py` — `Dataset` class: loads `*_prep.h5`, splits the weight column (always the **last**
     column of `gen_event_features`) out of the event features, **standardizes with hardcoded mean/std
     constants** (computed from a past reference run, not recomputed per-dataset), and normalizes sample
     weights. Supports MPI-style sharding via `rank`/`size` (Horovod ranks). Three behaviours here are
     easy to get wrong:
     - **Parameter features** — `param_names=[...]` plus `file_params=[{...}, ...]` (one dict per entry
       in `file_names`) appends each file's parameter values as extra event-feature columns, *after* the
       5 physics columns. They are excluded from standardization by padding `mean_event`/`std_event`
       with `0.0`/`1.0`, so `standardize()`/`revert_standardize()` pass them through untouched — no
       special-casing inside those methods. `file_params` is independent of the manifest, which is what
       lets the reference be labelled with *other* samples' parameter values.
     - **Weight normalization is per file** — each entry's weights are divided by that entry's own event
       count *and* by `len(file_names)`, then the whole array is scaled by `norm`. So a dataset's total
       weight is `norm × (mean of the per-file mean weights)`, independent of both file size and file
       count. Two datasets sharing a `norm` therefore always balance, including per parameter point.
       `self.nmax` is the **total** event count across all entries (not the first file's, as it once was).
     - **A path repeated in `file_names` is split into disjoint chunks**, one per occurrence: occurrence
       *k* reads `[k*available, k*available + file_nmax)` where `available = file_total // n_copies`.
       This is how a sample is replicated across parameter values without any event appearing twice.
       When a path repeats, `nmax` becomes a per-copy *ceiling* capped at `available` (warned once on
       rank 0), not an exact count. Consequence: passing the same path twice can no longer mean
       "two identical copies".
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
     Reads `PARAMETERS` from the config into `self.param_names` and sets
     `self.num_event = NEVT + len(param_names)` — `NEVT` in the config stays at the **physics** event
     feature count (5); the parameter columns are added on top. `self.num_feat` (particle-level, 7) is
     unrelated to the parameters and must not be bumped.
   - `train.py` — entry point. `data_files` is a hardcoded list — edit it to choose which parameter
     points to train on. The reference is **not** hardcoded: it comes from the manifest's
     `is_reference: true` entry and is replicated once per data file (see "Parameterized classifier").
     Must be launched under Horovod (`hvd.init()` is called unconditionally). On this SLURM cluster,
     launch multi-GPU runs with `srun`, not `horovodrun`.
   - `utils.py` — `LoadJson(file_name, base_path="../configs")`: despite the name and `.json` extension,
     config files are parsed with `yaml.safe_load`, not `json.load` — `configs/config_surrogate.json`
     actually uses YAML flow-mapping syntax (single-quoted keys, trailing comma), which is not valid
     JSON but is valid YAML. Because `base_path` defaults to `../configs`, training/analysis scripts
     assume they are launched **from inside `scripts/`**. Also holds the manifest helpers:
     `LoadManifest`, `GetReferenceFile` (the single `is_reference: true` entry; raises if not exactly
     one), `GetManifestEntry` (full record by path), and `GetFileParams` (list of parameter dicts for a
     list of paths, validated against `param_names`).

7. **Analysis / QA plotting**:
   - `scripts/analyze_classifier.py` — reconstructs the `Classifier`, loads checkpoint weights,
     evaluates `model_ema` on the "data" sample to get reweighting factors, reverts standardization, and
     produces the training loss curve plus per-feature (event- and particle-level) three-panel plots:
     unweighted data vs. model-reweighted data vs. reference, with a ratio-to-reference panel.
     `--data_file <manifest path>` picks which non-reference sample to reweight toward; the reference
     itself comes from the manifest's `is_reference` flag. **The reference is fed the _target's_
     parameter values**, not its own — that is the point at which the ratio is wanted, and it matches
     how the reference was labelled during training. Plot labels still show the reference's true
     physical parameter values (e.g. `alpha_s = 0.118`), built by `_format_param_label` from the
     manifest; `PARAM_LABELS` maps parameter names to LaTeX and `EVENT_NAMES` is extended at runtime so
     the parameter columns get labelled axes.
   - `scripts/QA_plots.py` — pre-training sanity-check plots comparing raw (unweighted) particle/event
     feature distributions across multiple Pythia alpha_s samples (and references some H1 real-data/
     Rapgap/Djangoh h5 paths under `/global/cfs/cdirs/m3246/H1/...` for future comparisons). Also
     defines an `hvd.init()` at import time. **Stale** — it passes `is_mc=True` to `Dataset`, which has
     never been a parameter, and its hardcoded `all_file_names` dict uses an older naming convention
     (`alphaS118`, `10mil`) that does not match the files on disk. Expect to fix it before use.
   - `plot_utils.py` (repo root) — a large (~2600 line) legacy plotting module referencing
     `Rapgap`/`Djangoh`/`options.name_translate`/unfolded weights — this is leftover from a different
     (unfolding-style) analysis and **is not imported by any current script in this repo**. Treat as
     dead/reference code, not part of the active pipeline.

## Parameterized classifier: the dataset manifest and reference replication

`configs/dataset_manifest.yaml` maps each `*_prep.h5` file to its physics parameter values. It is a
YAML list of records; paths are **filenames only**, paired at runtime with the `--data_folder` base
path, so the manifest stays portable across machines:

```yaml
- path: pythia_H1_alphaS0.1180_eplus_5Mevents_prep.h5
  parameters:
    alpha_s: 0.118
  is_reference: true          # exactly one entry carries this
- path: pythia_H1_alphaS0.1500_eplus_1Mevents_prep.h5
  parameters:
    alpha_s: 0.15
```

`parameters` is a dict, not a fixed column, so adding a second parameter later is just another key.
Which parameters actually reach the network is controlled by `PARAMETERS` in
`configs/config_surrogate.json` — the manifest may carry more than the model uses.

**Why the reference is replicated.** The reference is one fixed sample (alpha_s = 0.118), while the
data spans many parameter points. If the reference only ever carried its own `alpha_s = 0.118`, that
column would be a *perfect class label*: the classifier would learn "alpha_s == 0.118 → reference"
and drive the loss to zero without ever looking at the physics features, making the likelihood ratio
meaningless. So `train.py` lists the reference path once per data file and labels each copy with that
data file's parameter values:

```python
data_params     = utils.GetFileParams(manifest, data_files, surrogate.param_names)
reference_files = [utils.GetReferenceFile(manifest)["path"]] * len(data_files)
reference_params = data_params        # reference copies carry the DATA's parameter values
```

Because `Dataset` splits a repeated path into disjoint chunks, this costs no duplicated events: with
10 data points and a 5M-event reference, each point is paired against its own 500k reference events
and the whole file is used exactly once. That is equivalent to assigning every reference event one
parameter value — the standard parameterized-classifier construction — rather than making copies.
Since the reference distribution does not depend on the parameter and the network shares weights
across parameter values, the model still learns the reference from all 5M events; the split only
decides which parameter point each event is paired against.

Per-parameter-point weight balance falls out of the per-file normalization automatically: with
N data files and N reference chunks, both sides contribute `1/N` at every point.

## Data flow / directory layout

```
pythia_files/            raw Pythia ROOT output (pythia_H1.cc)              [gitignored]
clustered_pythia_files/  FastJet-clustered ROOT files (cluster_jets.py)     [gitignored, separate track]
pythia_h5/                *.h5          -> prepare_pythia_data.py output
                           *_prep.h5     -> preprocess_pythia.py output (network-ready)
configs/                 config_surrogate.json (YAML!) + dataset_manifest.yaml   [tracked]
weights/                 model checkpoints per MODEL_NAME + training history .pkl  [gitignored]
plots/, scripts/plots*/  QA and analysis output figures                    [gitignored]
```

`pythia_files/`, `pythia_h5/`, `clustered_pythia_files/`, `weights/`, `plots/`, compiled binaries
(`pythia_generation/pythia_H1`, `pythia_generation/cluster_centauro`), and `*.txt` logs are all
gitignored — this repo tracks code only, all data/model artifacts live on CFS and are regenerated by
the pipeline scripts above.

## Running the pipeline

The conda environment is defined by `environment.yml` (name `unbinned_inference`; on this machine it
lives at `/u/rmilton/.conda/envs/unbinned_inference`). Horovod is deliberately **not** in that file —
it needs build-time env vars, see `install_horovod.sh`. TensorFlow is pinned to `2.15.0.post1` to stay
on Keras 2 (`tf.keras`); 2.16+ defaults to Keras 3 and will break `architecture.py`. The jet-clustering
track additionally assumes CVMFS is available (`run_centauro.sh` sources the LCG_109
x86_64-el9-gcc15-opt view for `fastjet-config`/Centauro).

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

# Train the classifier (from scripts/, under Horovod via SLURM's srun)
srun --mpi=pmi2 python train.py --training_config config_surrogate.json

# Post-training analysis plots (--data_file picks which parameter point to reweight toward)
python analyze_classifier.py --data_folder <h5 dir> --weights_directory ../weights \
    --training_config config_surrogate.json --output_dir ./plots \
    --data_file pythia_H1_alphaS0.1500_eplus_1Mevents_prep.h5

# Pre-training QA comparison plots across raw samples (see staleness note above)
python QA_plots.py --mc_keys pythia_alphaS15 pythia_alphaS118
```

There is no test suite, linter, or CI configuration in this repo. To sanity-check dataloader changes,
build small synthetic `.h5` files (`gen_particle_features` `(N,132,7)` + `gen_event_features` `(N,6)`
with the weight last) in a scratch dir and assert on shapes/weights — that is how the parameter-feature
and disjoint-chunk behaviour above was verified. Watch out for float32: event IDs above ~2^24 are not
exactly representable and will look like duplicate events.

## Gotchas

- **Import order matters.** `matplotlib` and `dataloader` must be imported *before* TensorFlow/Horovod.
  The conda env's `scipy`/`matplotlib` binaries need a newer `libstdc++` than TF's bundled one, so
  importing TF first yields `GLIBCXX_3.4.30 not found` / `CXXABI_1.3.15 not found`. The scripts already
  order their imports correctly — don't "tidy" them. (`dataloader.py` used to import
  `sklearn.ensemble.GradientBoostingRegressor` unused, which tripped this; it has been removed.)
- **`configs/config_surrogate.json` is YAML, not JSON.** Editors will flag it as invalid JSON — that is
  a false positive; it is parsed with `yaml.safe_load`. Keep the single-quoted keys and trailing comma.
- **`NEVT` vs parameter count.** `NEVT` is the physics event-feature count (5). `SurrogateModel` adds
  `len(PARAMETERS)` on top. Do not bump `NEVT` when adding a parameter, and never bump `NFEAT` (that is
  the particle-level count, unaffected by parameters).
- **Reweighting uses `model_ema`**, not the raw trained model.
- HDF5 reads: the `*_prep.h5` datasets are contiguous and unchunked, so fancy/random row indexing is
  ~44x slower than a contiguous slice. Keep reads contiguous. Events are already in Pythia generation
  order, so any contiguous slice is already an unbiased sample — there is no need to shuffle on load.
