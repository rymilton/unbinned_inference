import argparse
import gc
import os

from dataloader import Dataset
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

hvd.init()

label_names = {
    "pythia_alphaS14": "Pythia $e^+, \\alpha_S=0.14$",
    "pythia_alphaS1136": "Pythia $e^+, \\alpha_S=0.1136$",
    "pythia_alphaS118": "Pythia $e^+, \\alpha_S=0.118$",
    "Djangoh_Eplus": "DJANGOH $e^+$",
    "Djangoh_Eminus": "DJANGOH $e^-$",
    "Rapgap_Eplus": "RAPGAP $e^+$",
    "Rapgap_Eminus": "RAPGAP $e^-$",
    "Rapgap_Eplus_sys0": "RAPGAP $e^+$ sys0",
    "Rapgap_Eplus_sys1": "RAPGAP $e^+$ sys1",
    "Rapgap_Eplus_sys5": "RAPGAP $e^+$ sys5",
    "Rapgap_Eplus_sys7": "RAPGAP $e^+$ sys7",
    "Rapgap_Eplus_sys11": "RAPGAP $e^-$ sys11",
    "Rapgap_Eminus_sys0": "RAPGAP $e^-$ sys0",
    "Rapgap_Eminus_sys1": "RAPGAP $e^-$ sys1",
    "Rapgap_Eminus_sys5": "RAPGAP $e^-$ sys5",
    "Rapgap_Eminus_sys7": "RAPGAP $e^-$ sys7",
    "Rapgap_Eminus_sys11": "RAPGAP $e^-$ sys11",
}

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config_general.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--plot_folder",
        default="./plots",
        help="Folder to store plots",
    )
    parser.add_argument(
        "--reco",
        action="store_true",
        default=False,
        help="Plot reco level results",
    )
    parser.add_argument(
        "--gen_only",
        action="store_true",
        default=False,
        help="Only load gen level results"
    )
    parser.add_argument(
        "--pass_gen",
        action="store_true",
        default=False,
        help="Apply pass_gen event mask when plotting.",
    )
    parser.add_argument(
        "--pass_reco",
        action="store_true",
        default=False,
        help="Apply pass_reco event mask when plotting.",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=1000000,
        help="Maximum number of events to load",
    )
    parser.add_argument(
        "--img_fmt",
        default="pdf",
        help="Format of the output figures",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Increase print level",
    )
    parser.add_argument(
        "--mc_keys",
        nargs="+",
        default=["pythia_alphaS14", "pythia_alphaS1136", "pythia_alphaS118", "Rapgap_Eplus", "Djangoh_Eplus"],
        help="MC samples to load. Must match keys in all_file_names.",
    )

    return parser.parse_args()

def setup_gpus(local_rank):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], "GPU")

def undo_standardizing(flags, dataloaders):
    # Undo preprocessing
    for mc in dataloaders:
        if flags.reco:
            dataloaders[mc].part, dataloaders[mc].event = dataloaders[
                mc
            ].revert_standardize(
                dataloaders[mc].reco[0],
                dataloaders[mc].reco[1],
                dataloaders[mc].reco[-1],
            )
            dataloaders[mc].mask = dataloaders[mc].reco[-1]
            del dataloaders[mc].reco
        else:
            dataloaders[mc].part, dataloaders[mc].event = dataloaders[
                mc
            ].revert_standardize(
                dataloaders[mc].gen[0], dataloaders[mc].gen[1], dataloaders[mc].gen[-1]
            )
            dataloaders[mc].mask = dataloaders[mc].gen[-1]
            del dataloaders[mc].gen

        gc.collect()



def get_dataloaders(flags, mc_file_names):
    dataloaders = {}

    for mc_key, mc_file in mc_file_names.items():
        mc_folder = os.path.dirname(mc_file)
        mc_name = os.path.basename(mc_file)

        if flags.reco:
            dataloaders[mc_key] = Dataset(
                [mc_name],
                mc_folder,
                is_mc=True,
                rank=hvd.rank(),
                size=hvd.size(),
                nmax=flags.nmax,
                pass_reco=flags.pass_reco,
            )
        else:
            dataloaders[mc_key] = Dataset(
                [mc_name],
                mc_folder,
                is_mc=True,
                rank=hvd.rank(),
                size=hvd.size(),
                nmax=flags.nmax,
                pass_fiducial=flags.pass_gen,
                gen_only=flags.gen_only,
            )

        gc.collect()

    return dataloaders

def gather_data(flags, dataloaders):
    for key in dataloaders:
        n_events = dataloaders[key].event.shape[0]

        event_plot_mask = np.ones(n_events, dtype=bool)

        if flags.pass_gen:
            if not hasattr(dataloaders[key], "pass_gen"):
                raise AttributeError(
                    f"{key} does not have pass_gen, but --pass_gen was requested."
                )
            event_plot_mask &= np.asarray(dataloaders[key].pass_gen, dtype=bool)

        if flags.pass_reco:
            if not hasattr(dataloaders[key], "pass_reco"):
                raise AttributeError(
                    f"{key} does not have pass_reco, but --pass_reco was requested."
                )
            event_plot_mask &= np.asarray(dataloaders[key].pass_reco, dtype=bool)

        # Event-level selection
        selected_event = dataloaders[key].event[event_plot_mask]

        # Particle-level selection:
        # dataloaders[key].mask has shape (n_events, n_particles). This mask is particle pT>0
        # event_plot_mask[:, None] applies the event cut to all particles in failed events.
        particle_plot_mask = dataloaders[key].mask & event_plot_mask[:, None]



        flat_part = dataloaders[key].part.reshape(
            (-1, dataloaders[key].part.shape[-1])
        )

        flat_particle_plot_mask = particle_plot_mask.reshape(-1)

        selected_part = flat_part[flat_particle_plot_mask]
        
        particle_weights = np.broadcast_to(
            dataloaders[key].weight[:, None],
            dataloaders[key].mask.shape,
        )

        flat_particle_weights = particle_weights.reshape(-1)
        selected_particle_weight = flat_particle_weights[flat_particle_plot_mask]

        dataloaders[key].part = hvd.allgather(
                tf.constant(selected_part)
            ).numpy()
        dataloaders[key].event = hvd.allgather(
                tf.constant(selected_event)
            ).numpy()
        selected_weight = dataloaders[key].weight[event_plot_mask]
        dataloaders[key].weight = hvd.allgather(
                tf.constant(selected_weight)
            ).numpy()
        dataloaders[key].particle_weight = hvd.allgather(
            tf.constant(selected_particle_weight)
        ).numpy()
        if flags.verbose and hvd.rank() == 0:
            print(
                f"{key}: gathered "
                f"{dataloaders[key].event.shape[0]} events and "
                f"{dataloaders[key].part.shape[0]} particles"
            )


            
def get_hist_binning(arrays, nbins):
    clean_arrays = []

    for arr in arrays:
        arr = np.asarray(arr)

        if arr.size == 0:
            continue

        arr = arr[np.isfinite(arr)]

        if arr.size > 0:
            clean_arrays.append(arr)

    if len(clean_arrays) == 0:
        return nbins

    values = np.concatenate(clean_arrays)

    if values.size == 0:
        return nbins

    return np.histogram_bin_edges(values, bins=nbins)


def plot_particles(flags, dataloaders, mc_keys, nbins=50):
    if hvd.rank() != 0:
        return

    particle_names = {
        "0": r"$\eta_p - \eta_e$",
        "1": r"$\phi_p - \phi_e - \pi$",
        "2": r"$\log(p_T)$",
        "3": r"$\log(p_T/Q)$",
        "4": r"$\log(E/Q)$",
        "5": r"$\log(E)$",
        "6": r"$\sqrt{(\eta_p - \eta_e)^2 + (\phi_p - \phi_e)^2}$",
        "7": "Absolute Charge",
    }
    binning_dict = {}


    os.makedirs(flags.plot_folder, exist_ok=True)

    first_key = mc_keys[0]
    n_features = dataloaders[first_key].part.shape[-1]

    for feature in range(n_features):
        plt.figure(figsize=(12, 8))

        values_for_binning = [
            dataloaders[key].part[:, feature]
            for key in mc_keys
        ]

        if feature not in binning_dict.keys():
            binning = get_hist_binning(values_for_binning, nbins)
        else:
            binning = binning_dict[feature]

        for key in mc_keys:
            values = dataloaders[key].part[:, feature]
            values = values[np.isfinite(values)]
            weights = dataloaders[key].particle_weight
            weights = weights[np.isfinite(values)]

            if values.size == 0:
                continue

            plt.hist(
                values,
                bins=binning,
                weights=weights,
                histtype="step",
                density=True,
                label=label_names[key],
            )

        if flags.reco and "data" in dataloaders:
            values = dataloaders["data"].part[:, feature]
            values = values[np.isfinite(values)]

            if values.size > 0:
                plt.hist(
                    values,
                    bins=binning,
                    histtype="step",
                    density=True,
                    label="data",
                )

        xlabel = particle_names.get(str(feature), f"Particle feature {feature}")
        plt.xlabel(xlabel)
        plt.ylabel("Normalized entries")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        append_string = "reco" if flags.reco else "gen"
        if flags.pass_gen:
            append_string += "_pass_gen"
        if flags.pass_reco:
            append_string += "_pass_reco"

        output_name = os.path.join(
            flags.plot_folder,
            f"part_{feature}_{append_string}.{flags.img_fmt}",
        )

        plt.savefig(output_name)
        plt.close()


def plot_event(flags, dataloaders, mc_keys, nbins=50):
    if hvd.rank() != 0:
        return

    event_names = {
        "0": r"$\log(Q^2)$",
        "1": r"$y$",
        "2": r"$e_{pT}/Q$",
        "3": r"$e_{\eta}$",
        "4": r"$e_{\phi}$",
    }
    if flags.pass_gen:
        binning_dict = {
            1: np.linspace(0.2,.7,51)
        }
    else:
        binning_dict = {
            1: np.linspace(0,1,51)
        }

    os.makedirs(flags.plot_folder, exist_ok=True)

    first_key = mc_keys[0]
    n_features = dataloaders[first_key].event.shape[-1]

    for feature in range(n_features):
        plt.figure(figsize=(12, 8))

        values_for_binning = [
            dataloaders[key].event[:, feature]
            for key in mc_keys
        ]
        if feature not in binning_dict:
            binning = get_hist_binning(values_for_binning, nbins)
        else:
            binning = binning_dict[feature]
        

        for key in mc_keys:
            values = dataloaders[key].event[:, feature]
            values = values[np.isfinite(values)]
            weights = dataloaders[key].weight
            weights = weights[np.isfinite(values)]

            if values.size == 0:
                continue

            plt.hist(
                values,
                bins=binning,
                weights=weights,
                histtype="step",
                density=True,
                label=label_names[key],
            )

        if flags.reco and "data" in dataloaders:
            values = dataloaders["data"].event[:, feature]
            values = values[np.isfinite(values)]

            if values.size > 0:
                plt.hist(
                    values,
                    bins=binning,
                    histtype="step",
                    density=True,
                    label="data",
                )

        xlabel = event_names.get(str(feature), f"Event feature {feature}")
        plt.xlabel(xlabel)
        plt.ylabel("Normalized entries")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        append_string = "reco" if flags.reco else "gen"
        if flags.pass_gen:
            append_string += "_pass_gen"
        if flags.pass_reco:
            append_string += "_pass_reco"

        output_name = os.path.join(
            flags.plot_folder,
            f"event_{feature}_{append_string}.{flags.img_fmt}",
        )

        plt.savefig(output_name)
        plt.close()


def main():

    """
    To-do:
    - Double check the get_dataloaders function -- this looks ok
    - Double check the undo_standardizing function -- this looks ok
    - Fix the gather_data function because it's terrible currently -- it now looks ok
    - Add the option to define your own binning -- ok done
    - Add ability to load and use model weights
    - Change to be able to use Pythia weights    
    """

    setup_gpus(hvd.local_rank())

    flags = parse_arguments()

    all_file_names = {
        "pythia_alphaS14": "/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/pythia_H1_alphaS14_eplus_10mil_prep.h5",
        "pythia_alphaS1136": "/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/pythia_H1_alphaS1136_eplus_10mil_prep.h5",
        "pythia_alphaS118": "/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/pythia_H1_alphaS118_eplus_10mil_prep.h5",
        "Djangoh_Eplus": "/global/cfs/cdirs/m3246/H1/April2026_h5/Djangoh_Eplus0607_prep.h5",
        "Djangoh_Eminus": "/global/cfs/cdirs/m3246/H1/April2026_h5/Djangoh_Eminus06_prep.h5",
        "Rapgap_Eplus": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_prep.h5",
        "Rapgap_Eminus": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_prep.h5",
        "Rapgap_Eplus_sys0": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_sys0_prep.h5",
        "Rapgap_Eplus_sys1": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_sys1_prep.h5",
        "Rapgap_Eplus_sys5": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_sys5_prep.h5",
        "Rapgap_Eplus_sys7": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_sys7_prep.h5",
        "Rapgap_Eplus_sys11": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eplus0607_sys11_prep.h5",
        "Rapgap_Eminus_sys0": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_sys0_prep.h5",
        "Rapgap_Eminus_sys1": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_sys1_prep.h5",
        "Rapgap_Eminus_sys5": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_sys5_prep.h5",
        "Rapgap_Eminus_sys7": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_sys7_prep.h5",
        "Rapgap_Eminus_sys11": "/global/cfs/cdirs/m3246/H1/April2026_h5/Rapgap_Eminus06_sys11_prep.h5",
    }
    # Models that we will use to get weights for each dataset at either gen or reco level
    model_paths = {
        "pythia_alphaS14": {"reco": [], "gen": []},
        "pythia_alphaS1136": {"reco": [], "gen": []},
        "pythia_alphaS118": {"reco": [], "gen": []},
        "Djangoh_Eplus": {"reco": [], "gen": []},
        "Djangoh_Eminus": {"reco": [], "gen": []},
        "Rapgap_Eplus": {"reco": {"Pretrain": "/global/cfs/cdirs/m3246/rmilton/H1Unfold_April2026_training/weights/OmniFold_pretrained_step1.pkl"}, "gen": {"Pretrain": "/global/cfs/cdirs/m3246/rmilton/H1Unfold_April2026_training/weights/OmniFold_pretrained_step2.pkl"}},
        "Rapgap_Eminus": {"reco": [], "gen": []},
        "Rapgap_Eplus_sys0": {"reco": [], "gen": []},
        "Rapgap_Eplus_sys1": {"reco": [], "gen": []},
        "Rapgap_Eplus_sys5": {"reco": [], "gen": []},
        "Rapgap_Eplus_sys7": {"reco": [], "gen": []},
        "Rapgap_Eplus_sys11": {"reco": [], "gen": []},
        "Rapgap_Eminus_sys0": {"reco": [], "gen": []},
        "Rapgap_Eminus_sys1": {"reco": [], "gen": []},
        "Rapgap_Eminus_sys5": {"reco": [], "gen": []},
        "Rapgap_Eminus_sys7": {"reco": [], "gen": []},
        "Rapgap_Eminus_sys11": {"reco": [], "gen": []},
    }

    missing_keys = [
        key for key in flags.mc_keys
        if key not in all_file_names
    ]

    if missing_keys:
        raise ValueError(
            f"Unknown MC keys: {missing_keys}. "
            f"Available keys are: {list(all_file_names.keys())}"
        )

    mc_files = {
        key: all_file_names[key]
        for key in flags.mc_keys
    }
    model_weights = {}

    for file_name in mc_files:
        models_to_evaluate = model_paths[file_name]["reco"] if flags.reco else model_paths[file_name]["gen"]
        if len(models_to_evaluate) == 0:
            continue
        model_weights[file_name] = {}
        print(models_to_evaluate)
        for model_name, model_path in models_to_evaluate.items():
            print(model_path)
            model_weights[file_name][model_name] = evaluate_model

    exit()

    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files: {mc_files}")

    dataloaders = get_dataloaders(flags, mc_files)
    model_weights = {}

    # Now we need to go through each dataloader and get the weights if there are any

    for file_name in mc_files:
        models_to_evaluate = model_paths[file_name]["reco"] if flags.reco else model_paths[file_name]["gen"]
        model_weights[file_name] = {}
        print(models_to_evaluate)
        for model_name, model_path in models_to_evaluate:
            print(model_path)
            model_weights[file_name][model_name] = []
    print(model_weights)

    undo_standardizing(flags, dataloaders)

    gather_data(flags, dataloaders)

    plot_particles(
        flags=flags,
        dataloaders=dataloaders,
        mc_keys=flags.mc_keys,
    )

    plot_event(
        flags=flags,
        dataloaders=dataloaders,
        mc_keys=flags.mc_keys,
    )


if __name__ == "__main__":
    main()