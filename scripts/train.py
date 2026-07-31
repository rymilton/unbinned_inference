import numpy as np
import argparse
from surrogate_model import SurrogateModel
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.backend as K
import utils
from dataloader import Dataset

# tf.random.set_seed(1234)
# np.random.seed(1234)


if __name__ == "__main__":
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="../pythia_h5/",
        help="Folder containing Pythia h5 files",
    )
    parser.add_argument(
        "--training_config",
        default="config_surrogate.json",
        help="Config file containing model training paramaters",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display additional information during training",
    )
    parser.add_argument(
        "--nmax",
        default=None,
        type=int,
        help="Number of events to load from each dataset",
    )
    parser.add_argument(
        "--weights_directory",
        default="../weights",
        type=str,
        help="Directory to store model",
    )


    flags = parser.parse_args()

    if flags.verbose:
        print(80 * "#")
        print(
            "Total hvd size {}, rank: {}, local size: {}, local rank{}".format(
                hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()
            )
        )
        print(80 * "#")



    data_files = ["pythia_H1_alphaS0.1100_eplus_1Mevents_prep.h5"]

    surrogate = SurrogateModel(
        version = "testing",
        config_file=flags.training_config,
        verbose=flags.verbose,
        weights_directory=flags.weights_directory,
    )

    manifest = utils.LoadManifest("dataset_manifest.yaml")
    data_params = utils.GetFileParams(manifest, data_files, surrogate.param_names)

    # Replicate the reference once per data file, each copy carrying that file's
    # parameter values. Without this the parameter columns alone would separate
    # reference from data and the classifier would never look at the physics.
    # Repeating the path makes Dataset split the reference into disjoint chunks.
    reference_files = [utils.GetReferenceFile(manifest)["path"]] * len(data_files)
    reference_params = data_params

    # Assume data_files will be many files, one for each parameter. Thus, we should standardize each one using their own mean and std
    data = Dataset(
        data_files,
        flags.data_folder,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=flags.nmax,
        param_names=surrogate.param_names,
        file_params=data_params,
    )

    reference = Dataset(
        reference_files,
        flags.data_folder,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=flags.nmax,
        norm = data.nmax,
        param_names=surrogate.param_names,
        file_params=reference_params,
    )
    exit()

    K.clear_session()

    surrogate.data = data
    surrogate.reference = reference
    surrogate.Preprocessing()
    surrogate.Classify()