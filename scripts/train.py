import numpy as np
import argparse
# from omnifold import Multifold
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.backend as K
# import utils
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
        default="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/",
        help="Folder containing Pythia h5 files",
    )
    parser.add_argument(
        "--config",
        default="config_general.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--config_omnifold",
        default="config_omnifold.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display additional information during training",
    )
    parser.add_argument(
        "--nmax",
        default=1_000_000,
        type=int,
        help="Number of events to load from each dataset",
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



    reference_files = ["pythia_H1_alphaS0.1180_eplus_100events_prep.h5"]
    data_files = ["pythia_H1_alphaS0.15_eplus_1000000events_prep.h5"]

    data = Dataset(
        data_files,
        flags.data_folder,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=flags.nmax,
    )

    reference = Dataset(
        reference_files,
        flags.data_folder,
        rank=hvd.rank(),
        size=hvd.size(),
        nmax=flags.nmax,
    )


    K.clear_session()