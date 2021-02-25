from ModelSpecification import NodeFormat, EdgeFormat, GlobalFormat, PositionFrame, \
    GraphAttributeFormat, GraphNetStructure, ModelSpecification, ClothKeypoints, TrainingParams, LossFunction


# Define a model specification which can be instantiated

def specify_input_graph_format() -> GraphAttributeFormat:
    return GraphAttributeFormat(
        node_format=NodeFormat.XYZR_FixedFlag,
        edge_format=EdgeFormat.DiffXYZ_ConnectionFlag,
        global_format=GlobalFormat.NextEndEffectorXYZR,
    )


def specify_position_output_graph_format() -> GraphAttributeFormat:
    return GraphAttributeFormat(
        node_format=NodeFormat.XYZ,
        edge_format=EdgeFormat.DiffXYZ,
        global_format=GlobalFormat.Dummy
    )


def specify_has_moved_output_graph_format() -> GraphAttributeFormat:
    return GraphAttributeFormat(
        node_format=NodeFormat.HasMovedClasses,
        edge_format=EdgeFormat.Dummy,
        global_format=GlobalFormat.Dummy
    )


def specify_cloth_keypoints_for_bag() -> ClothKeypoints:
    return ClothKeypoints(
        keypoint_indices=[
            # Front
            4, 127, 351, 380, 395, 557, 535, 550, 756, 783, 818, 1258,
            # Back
            150, 67, 420, 436, 920, 952, 1069, 1147, 1125, 1099, 929, 464,
            # Left
            142, 851, 1178,
            # Right
            49, 509, 1000,
            # Bottom
            641
        ],
        keypoint_edges=[
            # Front edges
            (4, 351), (4, 1258),
            (351, 380), (351, 818),
            (380, 395), (380, 783),
            (395, 756),
            (127, 557), (127, 1258),
            (557, 818), (557, 535),
            (535, 783), (535, 550),
            (550, 756),
            (783, 818),
            (818, 1258),
            # Back edges
            (436, 1069), (436, 420),
            (1069, 952),
            (952, 920),
            (420, 1099), (420, 464),
            (1099, 920), (1099, 1125),
            (920, 929),
            (464, 1125), (464, 67),
            (1125, 929), (1125, 1147),
            (67, 1147),
            (150, 1147), (150, 929),
            # Left edges
            (920, 1178),
            (1178, 535), (1178, 851),
            (150, 142),
            (851, 557), (851, 142), (851, 929),
            (142, 127),
            # Right edges
            (509, 380), (509, 420), (509, 1000),
            (1000, 351), (1000, 464), (1000, 49),
            (49, 4), (49, 67),
            # Bottom edges
            (641, 127), (641, 4),
            (641, 67), (641, 150),
        ],
        fixed_keypoint_indices=[395, 550, 756, 436, 952, 1069]
    )


def specify_graph_net_structure() -> GraphNetStructure:
    return GraphNetStructure(
        encoder_node_layers=[64, 64],
        encoder_edge_layers=[64, 64],
        encoder_global_layers=[128],
        core_node_layers=[128, 64],
        core_edge_layers=[128, 64],
        core_global_layers=[128],
        num_processing_steps=5,
    )


def specify_training_params() -> TrainingParams:
    return TrainingParams(
        frame_step=1,
        movement_threshold=0.001,
        batch_size=32,
    )


def specify_motion_model(name: str) -> ModelSpecification:
    return ModelSpecification(
        name=name,
        input_graph_format=specify_input_graph_format(),
        output_graph_format=specify_position_output_graph_format(),
        position_frame=PositionFrame.LocalToEndEffector,
        graph_net_structure=specify_graph_net_structure(),
        loss_function=LossFunction.MeanSquaredError_Position_NodesOnly,
        cloth_keypoints=specify_cloth_keypoints_for_bag(),
        training_params=specify_training_params(),
    )


def specify_has_moved_model(name: str) -> ModelSpecification:
    return ModelSpecification(
        name=name,
        input_graph_format=specify_input_graph_format(),
        output_graph_format=specify_has_moved_output_graph_format(),
        position_frame=PositionFrame.LocalToEndEffector,
        graph_net_structure=specify_graph_net_structure(),
        loss_function=LossFunction.CrossEntropy,
        cloth_keypoints=specify_cloth_keypoints_for_bag(),
        training_params=specify_training_params(),
    )


