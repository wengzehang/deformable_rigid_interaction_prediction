"""
New prediction models based on model specification
"""
from abc import ABC

import SimulatedData
import ModelSpecification
from ModelTrainer import ModelLoader
from PredictionInterface import PredictionInterface, PredictedFrame

from graph_nets import utils_tf

import numpy as np
import itertools
from typing import List


def create_input_graph_dict(specification: ModelSpecification.ModelSpecification,
                            current_frame: SimulatedData.Frame,
                            effector_xyzr_next: np.array,
                            hand_left_xyz_next: np.array,
                            hand_right_xyz_next: np.array):
    keypoint_indices = specification.cloth_keypoints.indices
    # The grasped keypoint indices are now taken from the dataset instead of being hardcoded
    # fixed_keypoint_indices = specification.cloth_keypoints.fixed_indices

    # Cloth object node features
    # Format: Position (XYZ), Radius (R), InverseDense flag (0 if fixed, 1 if movable)
    cloth_data_current = np.float32(current_frame.get_cloth_keypoint_info(keypoint_indices))

    # Rigid object node features
    # Format: Position (XYZ), Radius (R), InverseDense flag (always 1 for movable)
    rigid_data_current = np.float32(current_frame.get_rigid_keypoint_info())

    # Data for all nodes is stacked (cloth first, rigid second)
    node_data_current = np.vstack((cloth_data_current, rigid_data_current))

    # TensorFlow expects float32 values, the dataset contains float64 values
    effector_xyzr_current = np.float32(current_frame.get_effector_pose()).reshape(4)
    hand_left_xyz_current = current_frame.get_left_hand_position()
    hand_right_xyz_current = current_frame.get_right_hand_position()

    input_global_format = specification.input_graph_format.global_format
    input_global_features, current_position = input_global_format.compute_features(
        effector_xyzr_current, effector_xyzr_next,
        hand_left_xyz_current, hand_left_xyz_next,
        hand_right_xyz_current, hand_right_xyz_next)

    # Move to ModelSpecification.py?
    position_frame = specification.position_frame
    if position_frame == ModelSpecification.PositionFrame.Global:
        # No transformation needed
        pass
    elif position_frame == ModelSpecification.PositionFrame.LocalToEndEffector:
        # Transform positions to local frame (current effector position)
        node_data_current[:, :3] -= current_position
    else:
        raise NotImplementedError("Position frame not implemented")

    # Create input node features
    input_node_format = specification.input_graph_format.node_format

    # Next node data is only required for HasMovedClasses node format
    node_data_next = None
    movement_threshold = specification.training_params.movement_threshold
    input_node_features = input_node_format.compute_features(node_data_current,
                                                             node_data_current, node_data_next,
                                                             movement_threshold)

    input_edge_format = specification.input_graph_format.edge_format
    positions_current = node_data_current[:, :3]
    input_edge_features = input_edge_format.compute_features(positions_current,
                                                             specification.cloth_keypoints.keypoint_edges_from,
                                                             specification.cloth_keypoints.keypoint_edges_to)

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

    return input_graph_dict, current_position


class ModelFromSpecification(PredictionInterface, ABC):
    def __init__(self, model: ModelSpecification.ModelSpecification,
                 models_root_path: str = None):
        self.model_loader = ModelLoader(model, models_root_path)
        # We do not need to give example input data if we do not train the network
        self.model_loader.initialize_graph_net(None, None)

        # Recompute cloth edges (if loaded data is out of date)
        ck = self.model_loader.model.cloth_keypoints
        self.model_loader.model.cloth_keypoints = ModelSpecification.ClothKeypoints(ck.indices, ck.edges,
                                                                                    ck.fixed_indices)

    def create_input_graph_tuples(self, frame: SimulatedData.Frame,
                                  effector_xyzr_next: np.array,
                                  hand_left_xyz_next: np.array,
                                  hand_right_xyz_next: np.array):
        # Prepare input graph tuples
        input_graph_dict, current_position = create_input_graph_dict(self.model_loader.model,
                                                                     frame, effector_xyzr_next,
                                                                     hand_left_xyz_next, hand_right_xyz_next)
        input_graph_tuples = utils_tf.data_dicts_to_graphs_tuple([input_graph_dict])
        return input_graph_tuples, current_position

    def predict_output_graph_tuples(self, frame: SimulatedData.Frame,
                                    effector_xyzr_next: np.array,
                                    hand_left_xyz_next: np.array,
                                    hand_right_xyz_next: np.array):
        input_graph_tuples, current_position = self.create_input_graph_tuples(frame, effector_xyzr_next,
                                                                              hand_left_xyz_next, hand_right_xyz_next)

        # Model prediction
        predicted_graph_tuples = self.model_loader.compiled_predict(input_graph_tuples)
        return predicted_graph_tuples, current_position


class MotionModelFromSpecification(ModelFromSpecification):
    def __init__(self, model: ModelSpecification.ModelSpecification,
                 models_root_path: str = None):
        super().__init__(model, models_root_path)

        assert self.model_loader.model.output_graph_format.node_format == ModelSpecification.NodeFormat.XYZ, \
            "Output node format must be XYZ for the motion model"

    def predict_frame(self, frame: SimulatedData.Frame,
                      effector_xyzr_next: np.array,
                      hand_left_xyz_next: np.array,
                      hand_right_xyz_next: np.array
                      ) -> PredictedFrame:

        predicted_graph_tuples, current_position = self.predict_output_graph_tuples(frame, effector_xyzr_next,
                                                                                    hand_left_xyz_next,
                                                                                    hand_right_xyz_next)

        # Convert output graph tuple to PredictFrame
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        if self.model_loader.model.position_frame == ModelSpecification.PositionFrame.LocalToEndEffector:
            # Add the current position back to transform the center position to global coordinate
            # The position is dependent on the input GlobalFormat (either effector or hand position)
            predicted_nodes[:, :3] += current_position

        # The first entries are cloth keypoints (followed by rigid body nodes)
        num_keypoints = len(self.model_loader.model.cloth_keypoints.indices)
        cloth_keypoint_positions = predicted_nodes[:num_keypoints, :3]
        rigid_body_positions = predicted_nodes[num_keypoints:, :3]

        return PredictedFrame(cloth_keypoint_positions, rigid_body_positions)


def compute_moved_indices(predicted_nodes):
    has_moved_mask = np.argmax(predicted_nodes, axis=1)
    has_moved_indices = np.where(has_moved_mask > 0)
    return has_moved_indices


class HasMovedMaskModelFromSpecification(ModelFromSpecification):
    def __init__(self,
                 motion_model: PredictionInterface,
                 has_moved_spec: ModelSpecification.ModelSpecification,
                 models_root_path: str = None):
        super().__init__(has_moved_spec, models_root_path)

        self.motion_model = motion_model

        output_node_format = self.model_loader.model.output_graph_format.node_format
        assert output_node_format == ModelSpecification.NodeFormat.HasMovedClasses, \
            "Output node format must be HasMovedClasses for the has_moved mask model"

    def predict_frame(self, frame: SimulatedData.Frame,
                      effector_xyzr_next: np.array,
                      hand_left_xyz_next: np.array,
                      hand_right_xyz_next: np.array) -> PredictedFrame:
        # Predict motion without mask
        motion_prediction = self.motion_model.predict_frame(frame, effector_xyzr_next,
                                                            hand_left_xyz_next, hand_right_xyz_next)
        unmasked_pos_cloth = motion_prediction.cloth_keypoint_positions
        unmasked_pos_rigid = motion_prediction.rigid_body_positions

        # Predict has_moved mask
        predicted_graph_tuples, _ = self.predict_output_graph_tuples(frame, effector_xyzr_next,
                                                                     hand_left_xyz_next, hand_right_xyz_next)
        predicted_nodes = predicted_graph_tuples[-1].nodes.numpy()

        keypoint_indices = self.model_loader.model.cloth_keypoints.indices
        num_keypoints = len(keypoint_indices)

        # The first indices are the cloth keypoints (the last indices are rigid bodies)
        has_moved_cloth = compute_moved_indices(predicted_nodes[:num_keypoints])
        has_moved_rigid = compute_moved_indices(predicted_nodes[num_keypoints:])

        # Only override positions that have been classified as moving
        original_pos_cloth = frame.get_cloth_keypoint_positions(keypoint_indices)
        predicted_pos_cloth = original_pos_cloth
        predicted_pos_cloth[has_moved_cloth] = unmasked_pos_cloth[has_moved_cloth]

        original_pos_rigid = frame.get_rigid_keypoint_info()[:, :3]
        predicted_pos_rigid = original_pos_rigid
        predicted_pos_rigid[has_moved_rigid] = unmasked_pos_rigid[has_moved_rigid]

        return PredictedFrame(predicted_pos_cloth, predicted_pos_rigid)


class HorizonModel(PredictionInterface):
    def __init__(self,
                 single_prediction_model: ModelFromSpecification,
                 horizon_prediction_model: ModelFromSpecification,
                 start_horizon_frame: int = 1):

        # We use a base prediction model for frame-wise prediction
        self.single_prediction_model = single_prediction_model
        # And a longer horizon prediction model to reduce accumulation error
        self.horizon_prediction_model = horizon_prediction_model
        self.frame_step = horizon_prediction_model.model_loader.model.training_params.frame_step
        assert self.frame_step > 1, "Frame step must be greater than 1 for horizon prediction"

        # Start horizon prediction with this frame
        self.start_horizon_frame = start_horizon_frame

        # The horizon model was trained to predict 'frame_step' into the future
        # We use these predictions as anchor points for the frame-wise prediction
        # Anchor frames are filled in prepare_scenario()
        self.anchor_frames: List[PredictedFrame] = []

    def prepare_scenario(self, scenario: SimulatedData.Scenario):

        num_anchor_frames = scenario.num_frames() // self.frame_step
        anchor_frames = [None] * num_anchor_frames

        current_frame = scenario.frame(0)
        keypoint_indices = self.horizon_prediction_model.model_loader.model.cloth_keypoints.indices
        anchor_frames[0] = PredictedFrame(
            current_frame.get_cloth_keypoint_positions(keypoint_indices),
            current_frame.get_rigid_keypoint_info()[:, :3])
        next_frame = scenario.frame(self.frame_step)

        next_effector_position = next_frame.get_effector_pose()[0]
        hand_left_xyz_next = next_frame.get_left_hand_position()
        hand_right_xyz_next = next_frame.get_right_hand_position()
        prev_predicted_frame = self.horizon_prediction_model.predict_frame(current_frame, next_effector_position,
                                                                           hand_left_xyz_next, hand_right_xyz_next)

        for anchor_index in range(1, num_anchor_frames):
            current_frame = next_frame
            current_frame.overwrite_keypoint_positions(prev_predicted_frame.cloth_keypoint_positions)
            current_frame.overwrite_rigid_body_positions(prev_predicted_frame.rigid_body_positions)

            frame_index = anchor_index * self.frame_step
            next_frame = scenario.frame(frame_index + self.frame_step)
            next_effector_position = next_frame.get_effector_pose()[0]
            hand_left_xyz_next = next_frame.get_left_hand_position()
            hand_right_xyz_next = next_frame.get_right_hand_position()

            # Evaluate single frame
            predicted_frame = self.horizon_prediction_model.predict_frame(current_frame, next_effector_position,
                                                                          hand_left_xyz_next, hand_right_xyz_next)
            anchor_frames[anchor_index] = predicted_frame

            prev_predicted_frame = predicted_frame

        self.anchor_frames = anchor_frames

    def predict_frame(self, frame: SimulatedData.Frame,
                      effector_xyzr_next: np.array,
                      hand_left_xyz_next: np.array,
                      hand_right_xyz_next: np.array
                      ) -> PredictedFrame:

        anchor_index = frame.frame_index // self.frame_step
        anchor_frame: PredictedFrame = self.anchor_frames[anchor_index]

        # If we hit an anchor frame directly, we just return it
        anchor_frame_index = anchor_index * self.frame_step
        if anchor_frame_index >= self.start_horizon_frame and anchor_frame_index == frame.frame_index:
            return anchor_frame

        # Otherwise, we use the frame-wise prediction model
        return self.single_prediction_model.predict_frame(frame, effector_xyzr_next,
                                                          hand_left_xyz_next, hand_right_xyz_next)
