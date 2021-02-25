"""
Generate training data from
"""

import random
import itertools
from typing import Tuple
from sklearn.utils import shuffle
import numpy as np
from graph_nets import utils_tf

import SimulatedData
import ModelSpecification


class DataGenerator:
    def __init__(self,
                 data: SimulatedData.SimulatedData,
                 specification: ModelSpecification.ModelSpecification,
                 training: bool = True):
        self.data = data
        self.specification = specification
        self.training = training

        # We can generate a sample from each adjacent frame pair
        frame_step = specification.training_params.frame_step
        self.num_samples = data.num_scenarios * (data.num_frames - frame_step)

        # We generate scenario and frame index pairs which represent the input data for training
        # Each epoch, we randomly shuffle these indices to generate a new training order
        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, data.num_scenarios)
                        for frame_index in range(0, data.num_frames - frame_step)]
        self.indices = shuffle(self.indices)

        # Number of generated epochs (increased when the complete dataset has been generated/returned)
        self.epoch_count = 0

        # How many samples have been generated since the last epoch reset?
        self.generated_count = 0

        self.keypoint_edges_from = specification.cloth_keypoints.keypoint_edges_from
        self.keypoint_edges_to = specification.cloth_keypoints.keypoint_edges_to

    def next_batch(self, batch_size: int = None) -> Tuple:
        if batch_size is None:
            batch_size = self.specification.training_params.batch_size

        dataset_size = self.num_samples
        self.generated_count += batch_size
        if self.generated_count > dataset_size:
            self.indices = shuffle(self.indices)
            self.epoch_count += 1
            self.generated_count = 0
            return None, None, True

        start_index = random.randint(0, dataset_size - batch_size)
        end_index = start_index + batch_size

        input_dicts = [None] * batch_size
        target_dicts = [None] * batch_size

        frame_step = self.specification.training_params.frame_step

        batch_indices = self.indices[start_index:end_index]
        for i, (scenario_index, frame_index) in enumerate(batch_indices):
            scenario = self.data.scenario(scenario_index)
            current_frame = scenario.frame(frame_index)
            next_frame = scenario.frame(frame_index + frame_step)

            # input_dicts[i], target_dicts[i] = self.create_input_and_target_graph_dict(current_frame, next_frame)
            input_dicts[i], target_dicts[i] = self.create_input_and_target_graph_dict(current_frame, next_frame)

        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(input_dicts)
        target_graph_tuples = utils_tf.data_dicts_to_graphs_tuple(target_dicts)
        return input_graph_tuples, target_graph_tuples, False

    def create_input_and_target_graph_dict(self,
                                           current_frame: SimulatedData.Frame,
                                           next_frame: SimulatedData.Frame):
        keypoint_indices = self.specification.cloth_keypoints.indices

        # Cloth object node features
        # Format: Position (XYZ), Radius (R), InverseDense flag (0 if fixed, 1 if movable)
        cloth_data_current = np.float32(current_frame.get_cloth_keypoint_info(keypoint_indices))

        # Rigid object node features
        # Format: Position (XYZ), Radius (R), InverseDense flag (always 1 for movable)
        rigid_data_current = np.float32(current_frame.get_rigid_keypoint_info())

        # Data for all nodes is stacked (cloth first, rigid second)
        node_data_current = np.vstack((cloth_data_current, rigid_data_current))

        # Cloth object node features
        # Format: Position (XYZ), Radius (R), InverseDense flag (0 if fixed, 1 if movable)
        cloth_data_next = np.float32(next_frame.get_cloth_keypoint_info(keypoint_indices))

        # Rigid object node features
        # Format: Position (XYZ), Radius (R), InverseDense flag (always 1 for movable)
        rigid_data_next = np.float32(next_frame.get_rigid_keypoint_info())

        node_data_next = np.vstack((cloth_data_next, rigid_data_next))

        # TensorFlow expects float32 values, the dataset contains float64 values
        effector_xyzr_current = np.float32(current_frame.get_effector_pose()).reshape(4)
        effector_xyzr_next = np.float32(next_frame.get_effector_pose()).reshape(4)

        # Get hand positions
        hand_left_xyz_current = current_frame.get_left_hand_position()
        hand_left_xyz_next = next_frame.get_left_hand_position()
        hand_right_xyz_current = current_frame.get_right_hand_position()
        hand_right_xyz_next = next_frame.get_right_hand_position()

        input_global_format = self.specification.input_graph_format.global_format
        input_global_features, current_position = input_global_format.compute_features(
            effector_xyzr_current, effector_xyzr_next,
            hand_left_xyz_current, hand_left_xyz_next,
            hand_right_xyz_current, hand_right_xyz_next)

        output_global_format = self.specification.output_graph_format.global_format
        output_global_features, _ = output_global_format.compute_features(effector_xyzr_current, effector_xyzr_next,
                                                                          hand_left_xyz_current, hand_left_xyz_next,
                                                                          hand_right_xyz_current, hand_right_xyz_next)
        # Move to ModelSpecification.py?
        position_frame = self.specification.position_frame
        if position_frame == ModelSpecification.PositionFrame.Global:
            # No transformation needed
            pass
        elif position_frame == ModelSpecification.PositionFrame.LocalToEndEffector:
            # Transform positions to local frame (current effector position)
            new_origin = current_position
            node_data_current[:, :3] -= new_origin
            node_data_next[:, :3] -= new_origin
        else:
            raise NotImplementedError("Position frame not implemented")

        # Add random rotation to the position data (only during training)
        # zehang: if we do rotation for the input frame, we also need to rotate the effector future pose,
        #  and the ground truth, before calculating the edge attribute
        augment_rotation = hasattr(self.specification.training_params, 'augment_rotation') and \
                           self.specification.training_params.augment_rotation
        if position_frame == ModelSpecification.PositionFrame.LocalToEndEffector and \
                self.training and augment_rotation:
            # because we set the effector starting point as origin, we do rotation w.r.t. the verticle axis
            RYMat = SimulatedData.RandomRotateY_matrix()
            # rotate the effector future pose, the global features
            input_global_features[:3] = np.dot(input_global_features[:3], RYMat)
            # rotate the current frame
            node_data_current[:, :3] = np.dot(node_data_current[:, :3], RYMat)
            # rotate the output frame
            node_data_next[:, :3] = np.dot(node_data_next[:, :3], RYMat)

        movement_threshold = self.specification.training_params.movement_threshold
        # Create output node features (before applying noise)
        output_node_format = self.specification.output_graph_format.node_format
        output_node_features = output_node_format.compute_features(node_data_next,
                                                                   node_data_current, node_data_next,
                                                                   movement_threshold)

        output_edge_format = self.specification.output_graph_format.edge_format
        output_edge_features = output_edge_format.compute_features(node_data_next,
                                                                   self.specification.cloth_keypoints.keypoint_edges_from,
                                                                   self.specification.cloth_keypoints.keypoint_edges_to)

        # Add input noise to the position data (only during training)
        positions_current = node_data_current[:, :3]
        noise_stddev = self.specification.training_params.input_noise_stddev
        if self.training and noise_stddev is not None:
            noise = np.random.normal([0.0, 0.0, 0.0], noise_stddev, positions_current.shape)
            positions_current += noise

        # Create input node features (after applying noise)
        input_node_format = self.specification.input_graph_format.node_format
        input_node_features = input_node_format.compute_features(node_data_current,
                                                                 node_data_current, node_data_next,
                                                                 movement_threshold)

        input_edge_format = self.specification.input_graph_format.edge_format
        input_edge_features = input_edge_format.compute_features(positions_current,
                                                                 self.specification.cloth_keypoints.keypoint_edges_from,
                                                                 self.specification.cloth_keypoints.keypoint_edges_to)

        num_nodes = node_data_current.shape[0]
        edge_index = [i for i in itertools.product(np.arange(num_nodes), repeat=2)]
        # all connected, bidirectional
        node_edges_to, node_edges_from = list(zip(*edge_index))

        input_graph_dict = {
            "globals": input_global_features,
            "nodes": input_node_features,
            "edges": input_edge_features,
            "senders": node_edges_from,
            "receivers": node_edges_to,
        }

        output_graph_dict = {
            "globals": output_global_features,
            "nodes": output_node_features,
            "edges": output_edge_features,
            "senders": node_edges_from,
            "receivers": node_edges_to,
        }

        return input_graph_dict, output_graph_dict
