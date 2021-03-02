"""Visualization for simulated data
python DataVisualizer.py --task_index 1
"""
import argparse
import open3d
import numpy as np
from typing import List, Union, Tuple

import SimulatedData
import Datasets

class Frame:
    def __init__(self, mesh_tx_list: List[Union[open3d.geometry.TriangleMesh, open3d.geometry.PointCloud]],
                 cloth_mesh: open3d.geometry.TriangleMesh, cloth_pcd: open3d.geometry.PointCloud,
                 cloth_pcd_colors: np.array):
        self.mesh_tx_list = mesh_tx_list
        self.cloth_mesh = cloth_mesh
        self.cloth_pcd = cloth_pcd
        self.cloth_pcd_colors = cloth_pcd_colors

    def render(self, o3d_vis: open3d.visualization.Visualizer, reset_bounding_box=False):
        o3d_vis.clear_geometries()
        for geom in self.mesh_tx_list:
            o3d_vis.add_geometry(geom, reset_bounding_box)

    def update_keypoint(self, old_index: int, new_index: int, was_keypoint: bool,
                        o3d_vis: open3d.visualization.Visualizer):
        print("Changing keypoint:", old_index, "->", new_index)
        self.cloth_pcd_colors[old_index] = np.array([1.0, 1.0, 0.0]) if was_keypoint else np.array([0.0, 0.0, 0.0])
        self.cloth_pcd_colors[new_index] = np.array([1.0, 0.0, 0.0])
        self.cloth_pcd.colors = open3d.utility.Vector3dVector(self.cloth_pcd_colors)
        o3d_vis.update_geometry(self.cloth_pcd)

class DataVisualizer:
    # The video_id/visid is the scenario index, i.e. a single task execution
    def __init__(self, data: SimulatedData, scenario_index: int,
                 keypoint_indices: List[int], keypoint_edges: List[Tuple[int, int]], useeffector: int):
        self.data = data
        self.scenario_index = scenario_index
        self.frame_index = 0
        self.keypoint_index = 0
        shape = data.dataset[SimulatedData.MESH_KEY].shape
        self.num_scenarios = shape[0]
        self.num_frames = shape[1]
        self.num_mesh_points = shape[2]
        self.running = True
        self.show_cloth_mesh = True

        self.keypoint_indices = keypoint_indices
        self.keypoint_edges = keypoint_edges
        self.keypoint_edges_indices = [(keypoint_indices.index(f), keypoint_indices.index(t))
                                       for (f, t) in keypoint_edges]

        self.useeffector = useeffector
        self.frames = self.load_frames()

    def load_frames(self) -> List[Frame]:
        return [self.create_frame(i) for i in range(self.num_frames)]

    def create_frame(self, frame_index: int):

        dataset = self.data.dataset

        num_rigid = dataset[SimulatedData.RIGID_NUM_KEY][self.scenario_index]
        cloth_id = dataset[SimulatedData.CLOTH_ID_KEY][self.scenario_index]
        seq_rigid = dataset[SimulatedData.RIGID_KEY][self.scenario_index, frame_index, :num_rigid, :]  # (numrigid, 4), xyzr

        mesh_tx_list = []
        for obj_i in range(num_rigid):
            # get the origin of each rigid object
            xyz = seq_rigid[obj_i][0:3]
            # get the radius of each rigid object
            r = seq_rigid[obj_i][3]
            # create the sphere in open3d
            mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=r)
            # translate the sphere object according to the origin position
            mesh_tx = mesh_sphere.translate(xyz)
            # mesh_tx.compute_vertex_normals()
            # mesh_tx.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_tx_list.append(mesh_tx)

        seq = dataset[SimulatedData.MESH_KEY][self.scenario_index, frame_index, :]  # (numpoint, 3)
        # g1 = copy.copy(seq[:, 0])
        # g2 = copy.copy(seq[:, 2])
        # seq[:,2] = g1
        # seq[:,0] = g2
        conn = self.data.topodict[cloth_id]
        cloth_mesh = open3d.geometry.TriangleMesh()
        cloth_mesh.vertices = open3d.utility.Vector3dVector(seq)
        cloth_mesh.triangles = open3d.utility.Vector3iVector(conn)
        cloth_mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((seq.shape[0], 3), 0.5))

        if self.show_cloth_mesh:
            mesh_tx_list.append(cloth_mesh)

        #######################
        total_points = seq
        cloth_pcd = open3d.geometry.PointCloud()
        color_point = np.full(total_points.shape, np.array([0.0, 0.0, 0.0]))
        for i in self.keypoint_indices:
            color_point[i] = np.array([0.8, 0.0, 0.0])
        # Highlight grasped keypoints
        # TODO: We should get these from the loaded data and not hard-code them here
        color_point[756] = np.array([0.8, 0.8, 0.0])
        color_point[1069] = np.array([0.0, 0.8, 0.8])
        color_point[self.keypoint_index] = np.array([1.0, 0.0, 0.0])

        # color_point[[395, 550, 756, 436, 952, 1082]] = np.array([0.0, 0.8, 0.0])

        if self.show_cloth_mesh:
            cloth_pcd.points = open3d.utility.Vector3dVector(total_points)
            cloth_pcd.colors = open3d.utility.Vector3dVector(color_point)
        else:
            cloth_pcd.points = open3d.utility.Vector3dVector(total_points[self.keypoint_indices])
            cloth_pcd.colors = open3d.utility.Vector3dVector(color_point[self.keypoint_indices])
        mesh_tx_list.append(cloth_pcd)

        ##########################
        # effector

        if useeffector ==  Datasets.EffectorMotion.Ball:
            seq_effector = dataset[SimulatedData.EFFECTOR_KEY][self.scenario_index, frame_index, :][0]
            effector_xyz = seq_effector[0:3]
            effector_r = seq_effector[3]
            mesh_sphere_effector = open3d.geometry.TriangleMesh.create_sphere(radius=effector_r)
            # translate the sphere object according to the origin position
            mesh_tx_effector = mesh_sphere_effector.translate(effector_xyz)
            # mesh_tx_effector = mesh_sphere_effector
            mesh_tx_effector.paint_uniform_color([0.1, 0.1, 0.7])
            # mesh_tx.compute_vertex_normals()
            # mesh_tx.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_tx_list.append(mesh_tx_effector)

            total_points_e = seq_effector[0:3].reshape(-1, 3)
            pcd_e = open3d.geometry.PointCloud()
            pcd_e.points = open3d.utility.Vector3dVector(total_points_e)
            color_point_e = np.zeros(total_points_e.shape)
            pcd_e.colors = open3d.utility.Vector3dVector(color_point_e)
            mesh_tx_list.append(pcd_e)

        # Add line set between keypoints
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(seq[self.keypoint_indices])
        line_set.lines = open3d.utility.Vector2iVector(np.array(self.keypoint_edges_indices))
        line_set.colors = open3d.utility.Vector3dVector(np.full((len(self.keypoint_edges_indices), 3),
                                                                [1.0, 0.0, 1.0]))
        mesh_tx_list.append(line_set)

        return Frame(mesh_tx_list, cloth_mesh, cloth_pcd, color_point)

    def on_quit(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.running = False

    def on_prev_scenario(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.scenario_index = self.scenario_index - 1
        if self.scenario_index < 0:
            self.scenario_index = self.num_scenarios - 1
        self.frame_index = 0

    def on_next_scenario(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.scenario_index = self.scenario_index + 1
        if self.scenario_index >= self.num_scenarios:
            self.scenario_index = 0
        self.frame_index = 0

    def on_prev_frame(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.frame_index = self.frame_index - 1
        if self.frame_index < 0:
            self.frame_index = len(self.frames) - 1

    def on_next_frame(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.frame_index = self.frame_index + 1
        if self.frame_index >= len(self.frames):
            self.frame_index = 0

    def on_prev_keypoint(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.keypoint_index = self.keypoint_index - 1
        if self.keypoint_index < 0:
            self.keypoint_index = self.num_mesh_points - 1

    def on_next_keypoint(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.keypoint_index = self.keypoint_index + 1
        if self.keypoint_index >= self.num_mesh_points:
            self.keypoint_index = 0

    def on_toggle_cloth_mesh(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.show_cloth_mesh = not self.show_cloth_mesh

    def register_callbacks(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        GLFW_KEY_ESCAPE = 256
        GLFW_KEY_RIGHT = 262
        GLFW_KEY_LEFT = 263
        GLFW_KEY_DOWN = 264
        GLFW_KEY_UP = 265
        GLFW_KEY_A = 65
        GLFW_KEY_S = 83
        GLFW_KEY_SPACE = 32

        o3d_vis.register_key_callback(GLFW_KEY_ESCAPE, self.on_quit)
        o3d_vis.register_key_callback(GLFW_KEY_LEFT, self.on_prev_frame)
        o3d_vis.register_key_callback(GLFW_KEY_RIGHT, self.on_next_frame)
        o3d_vis.register_key_callback(GLFW_KEY_DOWN, self.on_prev_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_UP, self.on_next_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_A, self.on_prev_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_S, self.on_next_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_SPACE, self.on_toggle_cloth_mesh)

    def run(self):
        o3d_vis = open3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.create_window()
        o3d_vis.get_render_option().point_size = 10
        o3d_vis.get_render_option().line_width = 5
        o3d_vis.get_render_option().show_coordinate_frame = True

        self.register_callbacks(o3d_vis)

        old_scenario_index = -1
        old_frame_index = -1
        old_keypoint_index = -1
        old_show_cloth_mesh = self.show_cloth_mesh
        self.running = True
        while self.running:
            # running, scenario_index and frame_index are modified by the keypress events
            scenario_changed = self.scenario_index != old_scenario_index
            show_cloth_mesh_changed = self.show_cloth_mesh != old_show_cloth_mesh

            old_scenario_index = self.scenario_index
            if scenario_changed or show_cloth_mesh_changed:
                self.frames = self.load_frames()

            frame_changed = self.frame_index != old_frame_index
            keypoint_changed = self.keypoint_index != old_keypoint_index

            frame = self.frames[self.frame_index]
            if keypoint_changed:
                was_keypoint = old_keypoint_index in self.keypoint_indices
                frame.update_keypoint(old_keypoint_index, self.keypoint_index, was_keypoint, o3d_vis)
                old_keypoint_index = self.keypoint_index

            if scenario_changed or frame_changed or keypoint_changed or show_cloth_mesh_changed:
                # Only reset the bounding box if we load a new scene
                print("Rendering frame: ", self.scenario_index, ":", self.frame_index)
                frame.render(o3d_vis, reset_bounding_box=scenario_changed)
                old_frame_index = self.frame_index
                old_show_cloth_mesh = self.show_cloth_mesh

            o3d_vis.poll_events()
            o3d_vis.update_renderer()

        o3d_vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the training dataset for specific task.')
    parser.add_argument('--task_index', help='Specify the dataset id you want to visualize.',
                        type=int, default=12)
    parser.add_argument('--set_name', type=str, default=None)
    parser.add_argument('--demo', type=bool, default=False)
    # parser.add_argument('--frame_step', type=int, default=1)

    args, _ = parser.parse_known_args()

    subset = Datasets.Subset.from_name(args.set_name)
    if subset is None:
        raise ValueError(f"Subset with name '{args.set_name}' is unknown")

    # parse the directory to the dataset
    tasks_path = "./h5data/tasks"
    task = Datasets.get_task_by_index(args.task_index)

    task.isdemo = args.demo

    path_to_dataset = task.path_to_dataset(tasks_path, subset)
    path_to_topodict = task.path_to_topodict(tasks_path, subset)
    useeffector = task.effector_motion

    # load the data
    data = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)

    scenario_index = 0
    keypoint_indices = SimulatedData.keypoint_indices
    keypoint_edges = SimulatedData.keypoint_edges
    SimulatedData.validate_keypoint_graph(keypoint_indices, keypoint_edges)

    data_vis = DataVisualizer(data, scenario_index, keypoint_indices, keypoint_edges, useeffector)

    data_vis.run()
