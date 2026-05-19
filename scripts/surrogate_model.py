import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import gc
import horovod.tensorflow.keras as hvd
from scipy.special import expit
import logging
from architecture import Classifier
import utils
import pickle


class SurrogateModel:
    def __init__(
        self,
        version="Closure",
        config_file="config_omnifold.json",
        verbose=1,
        start=0,
        weights_directory="../weights",
    ):
        self.version = version
        self.verbose = verbose
        self.log_file = open("log_{}.txt".format(self.version), "w")
        self.opt = utils.LoadJson(config_file)
        self.start = start
        self.train_frac = 0.8
        self.num_feat = self.opt["NFEAT"]
        self.num_event = self.opt["NEVT"]
        self.lr = float(self.opt["LR"])
        self.size = hvd.size()
        self.lr_factor = 1.0

        self.num_steps_per_epoch = None
        self.data = None
        self.reference = None
        self.weights=None

        self.weights_folder = weights_directory
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

    def Classify(self):
        self.BATCH_SIZE = self.opt["BATCH_SIZE"]
        self.EPOCHS = self.opt["EPOCHS"]
        self.model_name = self.opt["MODEL_NAME"]

        # The ultimate goal of this model will be to train on many datasets simultaneously, but for now just assume we only have two datasets
        self.CompileModel(self.lr)
        self.RunClassification()
        self.CompileModel(self.lr, fixed=False)

    def RunClassification(self):
        """Data versus reco MC reweighting"""
        if hvd.rank() == 0:
            print("RUNNING CLASSIFICATION")

        self.RunModel(
            np.concatenate(
                (
                    self.labels_data,
                    self.labels_reference,
                )
            ),
            np.concatenate(
                (
                    self.data.weight, # TO-DO: Should come up with a better name than this
                    self.reference.weight
                )
            ),
            self.model,
            NTRAIN=self.num_steps_per_epoch * self.BATCH_SIZE,
            cached=False,  # after first training cache the training data
        )

        self.weights = self.reweight(
            self.data, self.model_ema, batch_size=1000
        )

    def RunModel(
        self,
        labels,
        weights,
        model,
        NTRAIN=1000,
        cached=False,
    ):
        test_frac = 1.0 - self.train_frac
        NTEST = int(test_frac * NTRAIN)
        train_data, test_data = self.cache(
            labels, weights, cached, NTRAIN - NTEST
        )

        if self.verbose and hvd.rank() == 0:
            print(80 * "#")
            self.log_string(
                "Train events used: {}, Test events used: {}".format(NTRAIN, NTEST)
            )
            print(80 * "#")

        verbose = 1 if hvd.rank() == 0 else 0

        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(
                patience=1000, min_lr=1e-7, verbose=verbose, monitor="val_loss"
            ),
            EarlyStopping(
                patience=self.opt["NPATIENCE"],
                restore_best_weights=True,
                monitor="val_loss",
            ),
        ]

        if hvd.rank() == 0:
            model_name = f"{self.weights_folder}/{self.model_name}/checkpoint"
            callbacks.append(
                ModelCheckpoint(
                    model_name,
                    save_best_only=True,
                    mode="auto",
                    period=1,
                    save_weights_only=True,
                )
            )

        hist = model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(self.train_frac * NTRAIN // self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=NTEST // self.BATCH_SIZE,
            verbose=verbose,
            callbacks=callbacks,
        )

        if hvd.rank() == 0:
            with open(model_name.replace("/checkpoint", ".pkl"), "wb") as f:
                pickle.dump(hist.history, f)

        del train_data, test_data
        gc.collect()

    def cache(self, label, weights, cached, NTRAIN):
        if not cached:
            self.idx = np.arange(label.shape[0])
            np.random.shuffle(self.idx)

            self.tf_data = tf.data.Dataset.from_tensor_slices(
                {
                    "inputs_particle": np.concatenate(
                        (
                            self.data.gen[0],
                            self.reference.gen[0],
                        )
                    )[self.idx],
                    "inputs_event": np.concatenate(
                        (
                            self.data.gen[1],
                            self.reference.gen[1],
                        )
                    )[self.idx],
                    "inputs_mask": np.concatenate(
                        (
                            self.data.gen[2],
                            self.reference.gen[2],
                        )
                    )[self.idx],
                }
            )
                # del self.mc.reco, self.data.reco
                # gc.collect()
            
        idx = self.idx

        if hvd.rank() == 0:
            print(label[idx])
            print(NTRAIN, idx.shape[0])
        labels = tf.data.Dataset.from_tensor_slices(
            np.stack((label[idx], weights[idx]), axis=1)
        )

        data = tf.data.Dataset.zip((self.tf_data, labels))

        train_data = (
            data.take(NTRAIN)
            .shuffle(NTRAIN)
            .repeat()
            .batch(self.BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        test_data = (
            data.skip(NTRAIN)
            .repeat()
            .batch(self.BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        del data
        gc.collect()
        return train_data, test_data

    def Preprocessing(self):
        self.PrepareInputs()
        self.PrepareModel()

    def CompileModel(self, lr, fixed=False):
        if self.num_steps_per_epoch is None:
            self.num_steps_per_epoch = (
                int(0.7 * (self.mc.nmax + self.data.nmax))
                // hvd.size()
                // self.BATCH_SIZE
            )
            if hvd.rank() == 0:
                print(self.num_steps_per_epoch)

        lr_schedule_body = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr / self.lr_factor,
            warmup_target=lr * np.sqrt(self.size) / self.lr_factor,
            warmup_steps=3 * int(self.train_frac * self.num_steps_per_epoch),
            decay_steps=self.EPOCHS * int(self.train_frac * self.num_steps_per_epoch),
            alpha=1e-2,
        )

        lr_schedule_head = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            warmup_target=lr * np.sqrt(self.size),
            warmup_steps=3 * int(self.train_frac * (self.num_steps_per_epoch)),
            decay_steps=self.EPOCHS * int(self.train_frac * self.num_steps_per_epoch),
            alpha=1e-2,
        )

        min_learning_rate = 1e-5
        opt_head = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_head,
            weight_decay=0.0,
            beta_1=0.95,
            beta_2=0.99,
        )

        opt_head = hvd.DistributedOptimizer(opt_head)

        opt_body = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_body,
            weight_decay=0.0,
            beta_1=0.95,
            beta_2=0.99,
        )

        opt_body = hvd.DistributedOptimizer(opt_body)

        self.model.compile(opt_body, opt_head)

    def PrepareInputs(self):
        self.labels_data = np.zeros(len(self.data.weight), dtype=np.float32)
        self.labels_reference = np.ones(len(self.reference.weight), dtype=np.float32)


    def PrepareModel(self):
        # Will assume same number of features for simplicity
        if self.verbose:
            self.log_string(
                "Preparing model architecture with: {} particle features and {} event features".format(
                    self.num_feat, self.num_event
                )
            )

        # TO-DO: NEED TO LOOK INTO CLASSIFIER ARCHITECTURE
        self.model = Classifier(
            self.num_feat,
            self.num_event,
            num_heads=self.opt["NHEADS"],
            num_transformer=self.opt["NTRANSF"],
            projection_dim=self.opt["NDIM"],
        )

        self.model_ema = self.model.model_ema

    def reweight(self, events, model, batch_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        f = expit(model.predict(events, batch_size=batch_size, verbose=self.verbose)[0])
        weights = f / (1.0 - f)
        return np.nan_to_num(weights[:, 0], posinf=1)

    def CompareDistance(self, patience, min_distance, weights1, weights2):
        distance = np.mean((np.sort(weights1) - np.sort(weights2)) ** 2)

        print(80 * "#")
        self.log_string("Distance between weights: {}".format(distance))
        print(80 * "#")

        if distance < min_distance:
            min_distance = distance
            patience = 0
        else:
            print(80 * "#")
            print("Distance increased! before {} now {}".format(min_distance, distance))
            print(80 * "#")
            patience += 1
        return patience, min_distance

    def log_string(self, out_str):
        self.log_file.write(out_str + "\n")
        self.log_file.flush()
        print(out_str)
