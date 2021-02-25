"""
Model specification for unified training and prediction models
"""

import GraphNetworkModules

import sonnet as snt
import tensorflow as tf
import numpy as np

from enum import Enum
from typing import List, Tuple
import itertools


class NodeFormat(Enum):
    """
    Node attribute format

    Dummy: One float is used as a dummy attribute.
    XYZ: The position (XYZ)
    XYZR: The position (XYZ) and the radius (R)
    XYZR_FixedFlag: The position (XYZ), the radius (R) and a fixed flag
    HasMovedClasses: Two-class classification result ([1.0 0.0] has not moved, [0.0 1.0] has moved)
    """
    Dummy = 0

    XYZ = 10
    XYZR = 11
    XYZR_FixedFlag = 12

    HasMovedClasses = 20

    def size(self):
        switcher = {
            NodeFormat.Dummy: 1,
            NodeFormat.XYZ: 3,
            NodeFormat.XYZR: 4,
            NodeFormat.XYZR_FixedFlag: 5,
            NodeFormat.HasMovedClasses: 2,
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("NodeFormat is not handled in size() function:", self)
        else:
            return result

    def compute_features(self, data: np.array, current_data: np.array, next_data: np.array,
                         movement_threshold: float):
        if self == NodeFormat.Dummy:
            return np.zeros(data.shape[0], np.float32)
        elif self == NodeFormat.XYZ:
            return data[:, :3]
        elif self == NodeFormat.XYZR:
            return data[:, :4]
        elif self == NodeFormat.XYZR_FixedFlag:
            return data[:, :5]
        elif self == NodeFormat.HasMovedClasses:
            # See whether the nodes have moved or not
            positions_current = current_data[:, :3]
            positions_next = next_data[:, :3]
            positions_diff = np.linalg.norm(positions_next - positions_current, axis=-1).reshape(-1, 1)
            has_moved = positions_diff > movement_threshold
            has_not_moved = positions_diff <= movement_threshold
            # The has_moved label is [1.0, 0.0] if the node did not move and [0.0, 1.0] if it moved
            has_moved_label = np.hstack((has_not_moved, has_moved)).astype(np.float32)
            return has_moved_label
        else:
            raise NotImplementedError("")


class EdgeFormat(Enum):
    """
    Edge attribute format

    Dummy: One float is used as a dummy attribute.
    DiffXYZ: Position difference (XYZ) to the adjacent node.
    DiffXYZ_ConnectionFlag: Position difference (XYZ) and connection flag (indicates physical connection in deformable object)
    """
    Dummy = 0

    DiffXYZ = 10
    DiffXYZ_ConnectionFlag = 11

    def size(self):
        switcher = {
            EdgeFormat.Dummy: 1,
            EdgeFormat.DiffXYZ: 3,
            EdgeFormat.DiffXYZ_ConnectionFlag: 4,
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("EdgeFormat is not handled in size():", self)
        else:
            return result

    def compute_features(self, node_data: np.array, keypoint_edges_to, keypoint_edges_from) -> np.array:
        positions = node_data[:, :3]

        num_nodes = positions.shape[0]
        # A fully connected graph has #nodes^2 edges
        num_edges = num_nodes * num_nodes

        edge_index = [i for i in itertools.product(np.arange(num_nodes), repeat=2)]
        # all connected, bidirectional
        node_edges_to, node_edges_from = list(zip(*edge_index))
        node_edges_to = list(node_edges_to)
        node_edges_from = list(node_edges_from)

        # The distance between adjacent nodes are the edges
        diff_xyz_connected = np.zeros((num_edges, 4), np.float32)  # DISTANCE 3D, CONNECTION TYPE 1.
        diff_xyz_connected[:, :3] = positions[node_edges_to] - positions[node_edges_from]

        if self == EdgeFormat.Dummy:
            return np.zeros(diff_xyz_connected.shape[0], np.float32)
        elif self == EdgeFormat.DiffXYZ:
            return diff_xyz_connected[:, :3]
        elif self == EdgeFormat.DiffXYZ_ConnectionFlag:
            # Fill connection flag
            connected_indices = keypoint_edges_to * num_nodes + keypoint_edges_from
            diff_xyz_connected[connected_indices, 3] = 1.0  # denote the physical connection
            # Bidirectional
            connected_indices = keypoint_edges_from * num_nodes + keypoint_edges_to
            diff_xyz_connected[connected_indices, 3] = 1.0  # denote the physical connection

            return np.vstack(diff_xyz_connected)
        else:
            raise NotImplementedError("EdgeFormat is not handled in compute_features():", self)


class GlobalFormat(Enum):
    """
    Global attribute format

    Dummy: One float is used as a dummy attribute.
    NextEndEffectorXYZR: Next position (XYZ) of the end effector (ball) and radius (R).
    NextHandPositionXYZ: Next position (XYZ) of the hand holding part of the bag.
    TODO: Right hand, left hand?
    """
    Dummy = 0
    NextEndEffectorXYZR = 10
    NextHandPositionXYZ_Left = 20
    NextHandPositionXYZ_Right = 21

    def size(self):
        switcher = {
            GlobalFormat.Dummy: 1,
            GlobalFormat.NextEndEffectorXYZR: 4,
            GlobalFormat.NextHandPositionXYZ_Left: 3,
            GlobalFormat.NextHandPositionXYZ_Right: 3,
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("GlobalFormat is not handled in size() function:", self)
        else:
            return result

    def compute_features(self,
                         effector_xyzr_current: np.array, effector_xyzr_next: np.array,
                         hand_left_xyz_current: np.array, hand_left_xyz_next: np.array,
                         hand_right_xyz_current: np.array, hand_right_xyz_next: np.array
                         ):
        if self == GlobalFormat.Dummy:
            features = np.zeros(1, np.float32)

            current_position = effector_xyzr_current[:3]
        elif self == GlobalFormat.NextEndEffectorXYZR:
            effector_position_diff = effector_xyzr_next[:3] - effector_xyzr_current[:3]
            effector_radius = effector_xyzr_current[3]

            features = np.zeros(4, np.float32)
            features[:3] = effector_position_diff
            features[3] = effector_radius

            current_position = effector_xyzr_current[:3]
        elif self == GlobalFormat.NextHandPositionXYZ_Left:
            position_diff = hand_left_xyz_next - hand_left_xyz_current

            features = position_diff

            current_position = hand_left_xyz_current
        elif self == GlobalFormat.NextHandPositionXYZ_Right:
            position_diff = hand_right_xyz_next - hand_right_xyz_current

            features = position_diff

            current_position = hand_right_xyz_current
        else:
            raise NotImplementedError("Global format is not handled:", self)

        return features, current_position


class PositionFrame(Enum):
    """
    Frame used for position attributes.

    Global: The global frame.
    LocalToEndEffector: A frame local to the end effector position (ball).
    """
    Global = 0
    LocalToEndEffector = 1


class GraphAttributeFormat:
    def __init__(self,
                 node_format: NodeFormat = NodeFormat.Dummy,
                 edge_format: EdgeFormat = EdgeFormat.Dummy,
                 global_format: GlobalFormat = GlobalFormat.Dummy):
        self.node_format = node_format
        self.edge_format = edge_format
        self.global_format = global_format


class ClothKeypoints:
    def __init__(self,
                 keypoint_indices: List[int] = None,
                 keypoint_edges: List[Tuple[int, int]] = None,
                 fixed_keypoint_indices: List[int] = None):
        self.indices = keypoint_indices
        self.edges = keypoint_edges
        self.fixed_indices = fixed_keypoint_indices
        self.fixed_indices_positions = [self.indices.index(keypoint_index) for keypoint_index in self.fixed_indices]

        self.keypoint_edges_from = np.array([keypoint_indices.index(f) for (f, _) in keypoint_edges])
        self.keypoint_edges_to = np.array([keypoint_indices.index(t) for (_, t) in keypoint_edges])


class NodeActivationFunction(Enum):
    """
    Node activation function on a graph network.

    Linear: Linear activation function (used for regression)
    Softmax: Softmax activation function (used for classification)
    """
    Linear = 0
    Softmax = 1


class GraphNetStructure:
    def __init__(self,
                 encoder_edge_layers: List[int] = None,
                 encoder_node_layers: List[int] = None,
                 encoder_global_layers: List[int] = None,
                 core_edge_layers: List[int] = None,
                 core_node_layers: List[int] = None,
                 core_global_layers: List[int] = None,
                 num_processing_steps=5,
                 node_activation_function=NodeActivationFunction.Linear):
        self.encoder_edge_layers = encoder_edge_layers
        self.encoder_node_layers = encoder_node_layers
        self.encoder_global_layers = encoder_global_layers
        self.core_edge_layers = core_edge_layers
        self.core_node_layers = core_node_layers
        self.core_global_layers = core_global_layers
        self.num_processing_steps = num_processing_steps
        self.node_activation_function = node_activation_function


def loss_mse_position_nodes_only(target, outputs):
    losses = [
        tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output.nodes[:, :3])
        for output in outputs
    ]
    return tf.stack(losses)


def loss_mse_position_nodes_and_edges(target, outputs):
    losses = [
        tf.compat.v1.losses.mean_squared_error(target.nodes[:, :3], output.nodes[:, :3]) +
        tf.compat.v1.losses.mean_squared_error(target.edges[:, :3], output.edges[:, :3])
        for output in outputs
    ]
    return tf.stack(losses)


def loss_crossentropy(targets, outputs):
    cce = tf.keras.losses.CategoricalCrossentropy()
    losses = [
        cce(targets.nodes, output.nodes)
        for output in outputs
    ]
    return tf.stack(losses)


class LossFunction(Enum):

    MeanSquaredError_Position_NodesOnly = 0
    MeanSquaredError_Position_NodesAndEdges = 1

    CrossEntropy = 10

    def create(self):
        if self == LossFunction.MeanSquaredError_Position_NodesOnly:
            return loss_mse_position_nodes_only
        elif self == LossFunction.MeanSquaredError_Position_NodesAndEdges:
            return loss_mse_position_nodes_and_edges
        elif self == LossFunction.CrossEntropy:
            return loss_crossentropy
        else:
            raise NotImplementedError("LossFunction not handled in create()", self)


class TrainingParams:
    def __init__(self,
                 frame_step: int = 1,
                 movement_threshold: float = 0.001,
                 batch_size: int = 32,
                 input_noise_stddev: float = 0.002,
                 augment_rotation: bool = False,
                 learning_rate: float = 1.0e-4,
                 early_stopping_epochs: int = 10,
                 ):
        self.frame_step = frame_step
        self.movement_threshold = movement_threshold
        self.batch_size = batch_size
        self.input_noise_stddev = input_noise_stddev
        self.augment_rotation = augment_rotation
        self.learning_rate = learning_rate
        self.early_stopping_epochs = early_stopping_epochs


def snt_mlp(layers):
    return lambda: snt.Sequential([
        snt.nets.MLP(layers, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


def snt_softmax(size: int):
    return lambda: snt.nets.MLP(
        [size],
        activation=tf.nn.softmax,
        activate_final=True,
        name="node_output")


def create_node_output_function(node_format: NodeFormat):
    if node_format == NodeFormat.HasMovedClasses:
        return snt_softmax(node_format.size())
    else:
        # A linear function is used inside EncodeProcessDecode
        return None


class ModelSpecification:
    """
    Specification of a trainable model.

    This specification is sufficient to
    - (re-)create the graph network architecture
    - convert input data (SimulatedData) into input graphs for the network
    - convert ground-truth data (SimulatedData) into target graphs for training the network
    - convert the network output back into frame data (PredictedFrame)
    - ... (Maybe more)
    TODO: Implement training based on this model specification
        - a) Implement conversion of input data (SimulatedData) into the desired input graph format ==> DONE
        - b) Implement conversion of output data (GraphsTuple) into predicted frame (PredictedFrame)
            ==> This is not always possible (Has moved classification)
        - c) Create an EncodeProcessDecode architecture based on spec ==> DONE
        - d) Do training based on spec (Do we need hyper parameters here as well?)

    TODO: Implement evaluation based on this model specification
        - a) Load weights from saved checkpoints
        - b) Implement predict step with the loaded model
        - c)


    The specification consists of the following attributes:
    - input_graph_format: The frame data is converted into this input graph format (for training and prediction)
    - output_graph_format: The network output format (also needs to be generated from ground-truth data)
    - graph_net_structure: The structure of the Encode-Process-Decode architecture (layer sizes and output functions)
    """

    def __init__(self,
                 name: str = None,
                 input_graph_format: GraphAttributeFormat = None,
                 output_graph_format: GraphAttributeFormat = None,
                 position_frame: PositionFrame = PositionFrame.LocalToEndEffector,
                 graph_net_structure: GraphNetStructure = None,
                 loss_function: LossFunction = LossFunction.MeanSquaredError_Position_NodesOnly,
                 cloth_keypoints: ClothKeypoints = None,
                 training_params: TrainingParams = None,
                 ):
        self.name = name
        self.input_graph_format = input_graph_format
        self.output_graph_format = output_graph_format
        self.position_frame = position_frame
        self.graph_net_structure = graph_net_structure
        self.loss_function = loss_function
        self.cloth_keypoints = cloth_keypoints
        self.training_params = training_params

    def create_graph_net(self):
        return GraphNetworkModules.EncodeProcessDecode(
            name=self.name,
            make_encoder_edge_model=snt_mlp(self.graph_net_structure.encoder_edge_layers),
            make_encoder_node_model=snt_mlp(self.graph_net_structure.encoder_node_layers),
            make_encoder_global_model=snt_mlp(self.graph_net_structure.encoder_global_layers),
            make_core_edge_model=snt_mlp(self.graph_net_structure.core_edge_layers),
            make_core_node_model=snt_mlp(self.graph_net_structure.core_node_layers),
            make_core_global_model=snt_mlp(self.graph_net_structure.core_global_layers),
            num_processing_steps=self.graph_net_structure.num_processing_steps,
            edge_output_size=self.output_graph_format.edge_format.size(),
            node_output_size=self.output_graph_format.node_format.size(),
            global_output_size=self.output_graph_format.global_format.size(),
            node_output_fn=create_node_output_function(self.output_graph_format.node_format),
    )
