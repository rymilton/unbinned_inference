import numpy as np

import os
import h5py as h5
import gc
import pickle
from collections import Counter, defaultdict


class Dataset:
    def __init__(
        self,
        file_names,
        base_path,
        rank=0,
        size=1,
        nmax=None,
        norm=None,
        pass_fiducial=False,
        pass_reco=False,
        preprocess=True,
        param_names=None,
        file_params=None,
    ):
        self.rank = rank
        self.size = size
        self.base_path = base_path
        self.nmax = nmax
        self.preprocess = preprocess
        self.param_names = param_names or []

        # Preprocessing parameters

        self.mean_part = [
            0.0,
            0.0,
            -0.761949,
            -3.663438,
            -2.8690917,
            0.03239748,
            3.9436243,
        ]
        self.std_part = [
            1.0,
            1.0,
            1.0133458,
            1.03931,
            1.0040112,
            0.98908925,
            1.2256976,
        ]

        self.mean_event = [6.4188385, 0.3331013, 0.8914633, -0.8352072, -0.07296985]
        self.std_event = [0.97656405, 0.1895471, 0.14934653, 0.4191545, 1.734126]

        # Parameter-value features (e.g. alpha_s) are appended after the physics event
        # features and excluded from standardization via identity mean/std.
        self.mean_event = self.mean_event + [0.0] * len(self.param_names)
        self.std_event = self.std_event + [1.0] * len(self.param_names)

        self.prepare_dataset(file_names, pass_fiducial, pass_reco, file_params)
        self.normalize_weights(self.nmax if norm is None else norm)

    def normalize_weights(self, norm):
        # print("Total number of reco events {}".format(self.num_pass_reco))
        # Each file's weights were already divided by that file's own nmax and by
        # the file count in prepare_dataset, so here we just rescale the combined
        # array to `norm`.
        self.weight = (norm * self.weight).astype(np.float32)

    def standardize(self, new_p, new_e, mask):
        mask = new_p[:, :, 2] != 0
        p = mask[:, :, None] * (new_p - self.mean_part) / self.std_part
        e = (new_e - self.mean_event) / self.std_event
        return p, e, mask

    def revert_standardize(self, new_p, new_e, mask):
        p = new_p * self.std_part + self.mean_part
        e = new_e * self.std_event + self.mean_event
        return p * mask[:, :, None], e

    def concatenate(self, data_list):
        data_part1 = [item[0] for item in data_list]  # Extracting all (M, P, Q) arrays
        data_part2 = [item[1] for item in data_list]  # Extracting all (M, F) arrays

        # Concatenate along the first axis (N * M)
        concatenated_part1 = np.concatenate(data_part1, axis=0)
        concatenated_part2 = np.concatenate(data_part2, axis=0)
        mask = concatenated_part1[:, :, 2] != 0
        del data_list
        gc.collect()
        return concatenated_part1, concatenated_part2, mask
    

    def prepare_dataset(self, file_names, pass_fiducial, pass_reco, file_params=None):
        """Load h5 files containing the data. The structure of the h5 file should be
        gen_particle_features : p_pt,p_eta,p_phi (B,N,3)
        gen_event_features    : Q2, e_px, e_py, e_pz, weight (B,5)

        A path listed more than once in file_names is split into that many disjoint
        chunks, one per occurrence. This is how a sample is replicated across several
        parameter values (see file_params) without any event appearing twice.
        """
        if self.param_names:
            assert file_params is not None and len(file_params) == len(file_names), (
                "file_params must be given, one entry per file_names, when param_names is set"
            )

        self.num_pass_gen = 0
        self.weight = []
        self.pass_gen = []
        gen = []
        requested_nmax = self.nmax  # cap requested by the caller, or None for each file's full size
        self.nmax = 0  # accumulates the total (all-file) event count as files are loaded
        path_counts = Counter(file_names)  # how many chunks each path is split into
        copies_seen = defaultdict(int)  # which chunk of that path we are on
        for ifile, f in enumerate(file_names):
            if self.rank == 0:
                print("Loading file {}".format(f))

            with h5.File(os.path.join(self.base_path, f), "r") as hf:
                file_total = hf["gen_event_features"].shape[0]

            n_copies = path_counts[f]
            copy_index = copies_seen[f]
            print(f, n_copies, copy_index)
            copies_seen[f] += 1

            # Each occurrence gets its own slice of the file, so copies never overlap.
            available = file_total // n_copies
            file_nmax = available if requested_nmax is None else min(requested_nmax, available)
            if (
                self.rank == 0
                and copy_index == 0
                and requested_nmax is not None
                and requested_nmax > available
            ):
                print(
                    "[WARNING] {} is split into {} disjoint copies; capping to {} events "
                    "per copy (requested {})".format(f, n_copies, available, requested_nmax)
                )
            chunk_offset = copy_index * available
            print(chunk_offset)
            self.nmax += file_nmax

            print("Num events total: ", file_nmax)

            per_rank = (file_nmax + self.size - 1) // self.size  # ceiling division
            start = chunk_offset + self.rank * per_rank
            end = min(start + per_rank, chunk_offset + file_nmax)
            print(start, end)

            # Sum of weighted events for collisions passing the gen cuts

            with h5.File(os.path.join(self.base_path, f), "r") as hf:
                gen_p = hf["gen_particle_features"][start:end].astype(np.float32)
                gen_e = hf["gen_event_features"][start:end].astype(np.float32)

                # Normalize this file's weights by its own event count, so files of
                # different sizes contribute on a comparable per-event scale, and by
                # the number of files, so each file supplies 1/N of the dataset's
                # total weight. This makes the total independent of both file size
                # and file count, so two datasets sharing a `norm` stay balanced.
                weights = gen_e[:, -1] / (file_nmax * len(file_names))
                self.weight.append(weights)

            event_feats = gen_e[:, :-1]  # Excluding the event weights from this
            if self.param_names:
                params = file_params[ifile]
                param_vals = np.array(
                    [params[name] for name in self.param_names], dtype=np.float32
                )
                event_feats = np.concatenate(
                    [event_feats, np.tile(param_vals, (event_feats.shape[0], 1))], axis=1
                )
            gen.append((gen_p, event_feats))
        self.weight = np.concatenate(self.weight)
        if self.preprocess:
            self.gen = self.standardize(*self.concatenate(gen))
        else:
            self.gen = self.concatenate(gen)
        del gen
        gc.collect()
        assert not np.any(np.isnan(self.gen[0])), "ERROR: NAN in particle dataset"
        assert not np.any(np.isnan(self.gen[1])), "ERROR: NAN in event dataset"