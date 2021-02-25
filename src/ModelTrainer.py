"""
Training a model given by a specification
"""

import SimulatedData
from ModelSpecification import ModelSpecification, LossFunction
import ModelDataGenerator

from graph_nets import utils_tf

import sonnet as snt
import tensorflow as tf
import numpy as np
import tqdm

import pickle
import os


class TrainingState:
    def __init__(self,
                 model: ModelSpecification = None,
                 best_metrics_tr: dict = None,
                 best_metrics_val: dict = None
                 ):
        self.model = model
        self.best_metrics_tr = best_metrics_tr
        self.best_metrics_val = best_metrics_val


class ModelLoader:
    def __init__(self,
                 model: ModelSpecification = None,
                 models_root_path: str = "./models/"
                 ):
        model_path = os.path.join(models_root_path, model.name)
        self.model_path = model_path

        self.state_path = os.path.join(model_path, "state.pickle")
        if os.path.exists(self.state_path):
            print("State file exists:", self.state_path)
            # Model has already been created/trained ==> load state from file
            with open(self.state_path, 'rb') as file:
                self.state = pickle.load(file)
        else:
            # Model is new ==> create new state to save later
            self.state = TrainingState(model)

        self.model = self.state.model
        self.net = None
        self.compiled_predict = None
        self.compiled_update_step = None
        self.compiled_compute_outputs = None
        self.checkpoint_save_prefix = None
        self.checkpoint = None

    def initialize_graph_net(self, example_input_data, example_target_data):
        # We create the graph network and loss function based on the model specification
        self.net = self.model.create_graph_net()
        loss_function = self.model.loss_function.create()

        batch_size = self.model.training_params.batch_size
        learning_rate = self.model.training_params.learning_rate
        optimizer = snt.optimizers.Adam(learning_rate)

        num_processing_steps = self.model.graph_net_structure.num_processing_steps

        def net_predict(inputs):
            outputs = self.net(inputs)
            return outputs

        def net_compute_outputs(inputs, targets):
            outputs = self.net(inputs)
            losses_per_processing_step = loss_function(targets, outputs)
            loss = tf.math.reduce_sum(losses_per_processing_step) / num_processing_steps

            return outputs, loss

        def net_update_step(inputs, targets):
            with tf.GradientTape() as tape:
                outputs, loss = net_compute_outputs(inputs, targets)

            gradients = tape.gradient(loss, self.net.trainable_variables)
            optimizer.apply(gradients, self.net.trainable_variables)

            return outputs, loss

        # Get the input signature for that function by obtaining the specs
        if example_input_data is not None and example_target_data is not None:
            input_signature = [
                utils_tf.specs_from_graphs_tuple(example_input_data),
                utils_tf.specs_from_graphs_tuple(example_target_data)
            ]
            # Compile the update function using the input signature for speedy code.
            self.compiled_update_step = tf.function(net_update_step, input_signature=input_signature)

        self.compiled_compute_outputs = tf.function(net_compute_outputs, experimental_relax_shapes=True)
        self.compiled_predict = tf.function(net_predict, experimental_relax_shapes=True)

        # Checkpoint setup
        checkpoint_root = os.path.join(self.model_path, "checkpoints")
        checkpoint_name = "checkpoint"
        self.checkpoint_save_prefix = os.path.join(checkpoint_root, checkpoint_name)

        # Make sure the model path exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # TODO: Allow loading a specific checkpoint?
        self.checkpoint = tf.train.Checkpoint(module=self.net)
        latest = tf.train.latest_checkpoint(checkpoint_root)
        if latest is not None:
            print("Loading latest checkpoint: ", latest)
            self.checkpoint.restore(latest)
        else:
            print("No checkpoint found. Beginning training from scratch.")

    def save_state(self):
        self.checkpoint.save(self.checkpoint_save_prefix)

        with open(self.state_path, 'wb') as file:
            pickle.dump(self.state, file, protocol=pickle.HIGHEST_PROTOCOL)


class ModelTrainer(ModelLoader):
    def __init__(self,
                 model: ModelSpecification = None,
                 models_root_path: str = "./models/",
                 train_path_to_topodict: str = None,
                 train_path_to_dataset: str = None,
                 valid_path_to_topodict: str = None,
                 valid_path_to_dataset: str = None
                 ):

        super().__init__(model, models_root_path)

        # Load dataset
        train_data = SimulatedData.SimulatedData.load(train_path_to_topodict, train_path_to_dataset)
        valid_data = SimulatedData.SimulatedData.load(valid_path_to_topodict, valid_path_to_dataset)

        # Generators which transform the dataset into graphs for training the network
        self.train_generator = ModelDataGenerator.DataGenerator(train_data, self.model, training=True)
        self.valid_generator = ModelDataGenerator.DataGenerator(valid_data, self.model, training=False)

        # Get some example data that resembles the tensors that will be fed into update_step():
        batch_size = self.model.training_params.batch_size
        example_input_data, example_target_data, _ = self.train_generator.next_batch(batch_size=batch_size)

        super().initialize_graph_net(example_input_data, example_target_data)

    def train(self):

        # Recover last minimal validation loss
        min_validation_loss = 1.0e10
        if self.state.best_metrics_val is not None:
            min_validation_loss = self.state.best_metrics_val["loss"]

        # Do early stopping if the validation loss has not decreased for n epochs
        early_stopping_epochs = self.model.training_params.early_stopping_epochs
        epochs_without_improvement = 0

        while epochs_without_improvement < early_stopping_epochs:
            metrics_tr, metrics_val = self.train_epoch()

            loss_val = metrics_val["loss"]
            if loss_val < min_validation_loss:
                print("Saving new checkpoint since validation loss has decreased.\n",
                      "Before:", min_validation_loss, "After:", loss_val,
                      "Epochs without improvement:", epochs_without_improvement)
                min_validation_loss = loss_val
                self.state.best_metrics_tr = metrics_tr
                self.state.best_metrics_val = metrics_val

                super().save_state()

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

    def train_epoch(self):
        batch_size = self.model.training_params.batch_size

        compute_accuracy = self.model.loss_function == LossFunction.CrossEntropy

        train_accuracy = tf.keras.metrics.Accuracy()
        valid_accuracy = tf.keras.metrics.Accuracy()

        pbar_tr = tqdm.tqdm(total=self.train_generator.num_samples)
        # Training set
        losses_tr = []
        while True:
            inputs_tr, targets_tr, new_epoch = self.train_generator.next_batch(batch_size=batch_size)
            if new_epoch:
                break
            pbar_tr.update(batch_size)

            outputs_tr, loss_tr = self.compiled_update_step(inputs_tr, targets_tr)

            losses_tr.append(loss_tr.numpy())

            # print("Training Loss:", loss_tr.numpy())
            if compute_accuracy:
                train_accuracy.update_state(tf.argmax(targets_tr.nodes, axis=1),
                                            tf.argmax(outputs_tr[-1].nodes, axis=1))
        pbar_tr.close()

        epoch = self.train_generator.epoch_count

        metrics_tr = {"loss": np.mean(losses_tr)}
        if compute_accuracy:
            metrics_tr["accuracy"] = train_accuracy.result().numpy()
        print("Epoch", epoch, "Training:", metrics_tr)

        pbar_val = tqdm.tqdm(total=self.valid_generator.num_samples)
        # Compute metrics on validation set
        losses_val = []
        while True:
            inputs_val, targets_val, new_epoch = self.valid_generator.next_batch(batch_size=batch_size)
            if new_epoch:
                break
            pbar_val.update(batch_size)

            outputs_val, loss_val = self.compiled_compute_outputs(inputs_val, targets_val)

            losses_val.append(loss_val.numpy())
            if compute_accuracy:
                valid_accuracy.update_state(tf.argmax(targets_val.nodes, axis=1),
                                            tf.argmax(outputs_val[-1].nodes, axis=1))
        pbar_val.close()

        metrics_val = {"loss": np.mean(losses_val)}
        if compute_accuracy:
            metrics_val["accuracy"] = valid_accuracy.result().numpy()
        print("Epoch", epoch, "Validation:", metrics_val)

        return metrics_tr, metrics_val
