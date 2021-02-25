"""Loader for hdf5 data

This module contains function for loading the simulated data.
The simulated data contains ...

"""

import h5py
import pickle

import numpy as np
from typing import List, Tuple


# Dataset keys:
MESH_KEY = 'posCloth'
RIGID_KEY = 'posRigid'
CLOTH_ID_KEY = 'clothid'
RIGID_NUM_KEY = 'numRigid'
EFFECTOR_KEY = 'posEffector'
GRASPED_INDEX_LEFT = 'graspind_l' # The moving hand in the circular motion datasets
GRASPED_INDEX_RIGHT = 'graspind_r'
GRASPED_NUM_LEFT = 'graspnum_l'
GRASPED_NUM_RIGHT = 'graspnum_r'


class Frame:
    def __init__(self, data: 'SimulatedData', scenario_index: int, frame_index: int):
        assert scenario_index >= 0
        assert scenario_index < data.num_scenarios
        assert frame_index >= 0
        assert frame_index < data.num_frames

        self.data = data
        self.scenario_index = scenario_index
        self.frame_index = frame_index
        self.num_rigid = self.data.dataset[RIGID_NUM_KEY][self.scenario_index]
        self.overwritten_keypoint_pos = None
        self.overwritten_rigid_body_pos = None

    def get_cloth_keypoint_positions(self, indices):
        mesh_vertices = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        return mesh_vertices[indices]

    def overwrite_keypoint_positions(self, keypoint_positions: np.array):
        self.overwritten_keypoint_pos = keypoint_positions

    def overwrite_rigid_body_positions(self, rigid_body_positions: np.array):
        self.overwritten_rigid_body_pos = rigid_body_positions

    def get_grasped_indices(self):
        # Get grasped vertex indices
        num_grasped_vertices_left = self.data.dataset[GRASPED_NUM_LEFT][self.scenario_index][self.frame_index]
        grasped_vertex_indices_left = self.data.dataset[GRASPED_INDEX_LEFT][self.scenario_index][self.frame_index]
        grasped_vertex_indices_left = grasped_vertex_indices_left[:num_grasped_vertices_left]

        num_grasped_vertices_right = self.data.dataset[GRASPED_NUM_RIGHT][self.scenario_index][self.frame_index]
        grasped_vertex_indices_right = self.data.dataset[GRASPED_INDEX_RIGHT][self.scenario_index][self.frame_index]
        grasped_vertex_indices_right = grasped_vertex_indices_right[:num_grasped_vertices_right]

        return np.concatenate((grasped_vertex_indices_left, grasped_vertex_indices_right))

    def get_left_hand_position(self) -> np.array:
        grasped_vertex_indices = self.data.dataset[GRASPED_INDEX_LEFT][self.scenario_index][self.frame_index]
        # Which index should we use as the hand position?
        hand_index = grasped_vertex_indices[5]

        mesh_frame = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        return np.float32(mesh_frame[hand_index])

    def get_right_hand_position(self) -> np.array:
        grasped_vertex_indices = self.data.dataset[GRASPED_INDEX_RIGHT][self.scenario_index][self.frame_index]
        # Which index should we use as the hand position?
        hand_index = grasped_vertex_indices[5]

        mesh_frame = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        return np.float32(mesh_frame[hand_index])

    def get_cloth_keypoint_info(self, indices):
        mesh_frame = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        mesh_vertices = mesh_frame[indices]
        num_keypoints = mesh_vertices.shape[0]
        keypoint_radius = np.ones((num_keypoints, 1)) * 1e-5

        # Replace keypoint positions if they have been overwritten
        if self.overwritten_keypoint_pos is not None:
            mesh_vertices = self.overwritten_keypoint_pos

        grasped_vertex_indices = self.get_grasped_indices()

        inverse_dense = np.ones((mesh_frame.shape[0],1))
        inverse_dense[grasped_vertex_indices] = 0.0
        inverse_dense = inverse_dense[indices]

        mesh_vertices = np.hstack((mesh_vertices, keypoint_radius, inverse_dense))
        return mesh_vertices

    def get_cloth_keypoint_info_partgraph(self, indices, fix_indices):
        mesh_frame = self.data.dataset[MESH_KEY][self.scenario_index][self.frame_index]
        mesh_vertices = mesh_frame[indices]
        num_keypoints = mesh_vertices.shape[0]
        keypoint_radius = np.ones((num_keypoints, 1)) * 1e-5


        inversedense = np.ones((mesh_frame.shape[0],1))
        inversedense[fix_indices] = 0.0
        inversedense = inversedense[indices]

        mesh_vertices = np.hstack((mesh_vertices, keypoint_radius, inversedense))

        # calculate the mass center of the particles
        # use it as a node
        deformnode = mesh_vertices.mean(axis=0)
        rad = ((deformnode[:3] - mesh_vertices[:, :3]) ** 2).sum(axis=1).max() / 2
        deformnode[3] = rad
        deformnode[4] = 1.0 # free, nonfix point
        deformnode = deformnode.reshape(1,5)

        mesh_vertices = np.vstack((mesh_vertices, deformnode))
        return mesh_vertices

    def get_rigid_keypoint_info(self):
        rigid_vertices = np.copy(self.data.dataset[RIGID_KEY][self.scenario_index][self.frame_index][:self.num_rigid,:])
        if self.overwritten_rigid_body_pos is not None:
            rigid_vertices[:, :3] = self.overwritten_rigid_body_pos
        inversedense = np.ones((rigid_vertices.shape[0],1)) #  * 100
        rigid_vertices = np.hstack((rigid_vertices, inversedense))
        return rigid_vertices

    def get_effector_pose(self):
        # function to get effector info of current frame
        effector_pose = self.data.dataset[EFFECTOR_KEY][self.scenario_index][self.frame_index]
        return effector_pose

    def get_whole_edge(self):
        # TODO: consider the effector and all the rigid sphere as graph nodes
        # Return: a new list of edge connection containing rigid object
        pass


class Scenario:
    def __init__(self, data: 'SimulatedData', scenario_index: int):
        assert scenario_index >= 0
        assert scenario_index < data.num_scenarios

        self.data = data
        self.scenario_index = scenario_index

    def num_frames(self):
        return self.data.num_frames

    def frame(self, frame_index: int):
        return Frame(self.data, self.scenario_index, frame_index)


class SimulatedData:
    """This class contains the simulated data

    The topodict contains the topology description for each map.
    It is a dictionary mapping the cloth id to the faces of each cloth.
    The faces are represented as a 2D array with shape (#faces, 3) where
    each face is represented as 3 vertex indices.

    The dataset contains the simulation results:
    TODO: Explain relevant keys here
    """

    def __init__(self, dataset, topodict):
        self.dataset = dataset
        self.topodict = topodict

        # The 'posCloth' entry has shape ( #scenario_ids, #frames, #mesh_points, 4[xyz, r] )
        shape = self.dataset[MESH_KEY].shape
        self.num_scenarios = shape[0]
        self.num_frames = shape[1]
        self.num_mesh_points = shape[2]
        self.num_rigid_object = 0

        # The 'posEffector' entry has shape ( #scenario_ids, #frames, #1, 4[xyz, r]  )

    @staticmethod
    def load(path_to_topodict: str, path_to_dataset: str) -> 'SimulatedData':

        with open(path_to_topodict, 'rb') as pickle_file:
            topodict = pickle.load(pickle_file)

        dataset = h5py.File(path_to_dataset, 'r')
        return SimulatedData(dataset, topodict)

    def scenario(self, scenario_index: int) -> Scenario:
        return Scenario(self, scenario_index)


keypoint_indices = [
    # Front
    4, 127, 351, 380, 395, 557, 535, 550, 756, 783, 818, 1258,
    # Back
    150, 67, 420, 436, 920, 952, 1082, 1147, 1125, 1099, 929, 464,
    # Left
    142, 851, 1178,
    # Right
    49, 509, 1000,
    # Bottom
    641
]

fix_keypoint_indices = [395, 550, 756, 436, 952, 1082]
fix_keypoint_place = [4, 7, 8, 15, 17, 18]

keypoint_edges = [
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
    (436, 1082), (436, 420),
    (1082, 952),
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
]



def validate_keypoint_graph(indices: List[int], edges: List[Tuple[int, int]]):
    for (e_from, e_to) in edges:
        if e_from not in indices:
            print("from:", e_from, "to:", e_to, " from is not in indices")
        if e_to not in indices:
            print("from:", e_from, "to:", e_to, " to is not in indices")

def RandomRotateY_matrix(angle=None):
    """Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    """
    if angle == None:
        angle = np.random.uniform(0, 2*np.pi)

    ry_matrix = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    return ry_matrix

def RandomRotateY(inputPos, angle=None):
    """Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    """

    if angle == None:
        angle = np.random.uniform(0, 2*np.pi)

    ry_matrix = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    return np.dot(inputPos, ry_matrix)
