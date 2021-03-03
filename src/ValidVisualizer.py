"""Visualization for simulated data

"""

from NewPredictionModels import *
import open3d
from datetime import datetime, timedelta
import numpy as np
from typing import List, Union, Tuple

# from SimulatedData import SimulatedData, Frame, keypoint_indices, keypoint_edges, MESH_KEY
from SimulatedData import *

# from DataVisualizer import Frame
import DataVisualizer
from LineMesh import LineMesh

import Datasets
import copy
import argparse
import tqdm


from Evaluation import create_prediction_model

class KeypointDataVisualizer:
    # The video_id/visid is the scenario index, i.e. a single task execution
    def __init__(self, data: SimulatedData, scenario_index: int,
                 keypoint_indices: List[int], keypoint_edges: List[Tuple[int, int]], useeffector: int):
        self.data = data
        self.scenario_index = scenario_index
        self.frame_index = 0
        self.keypoint_index = 0
        self.playing_video = False
        shape = data.dataset[MESH_KEY].shape
        self.num_scenarios = shape[0]
        self.num_frames = shape[1]
        self.num_mesh_points = shape[2]
        self.running = True
        self.show_cloth_mesh = True

        self.keypoint_indices = keypoint_indices
        self.keypoint_edges = keypoint_edges
        self.keypoint_edges_indices = [(keypoint_indices.index(f), keypoint_indices.index(t))
                                       for (f, t) in keypoint_edges]

        self.dataset_cloth = self.data.dataset[MESH_KEY][:]
        self.dataset_rigid = self.data.dataset[RIGID_KEY][:]
        self.useeffector = useeffector
        self.frames = self.load_frames()


    def load_frames(self) -> List[DataVisualizer.Frame]:
        return [self.create_frame(i) for i in range(self.num_frames)]

    def create_frame(self, frame_index: int):

        dataset = self.data.dataset

        num_rigid = dataset[RIGID_NUM_KEY][self.scenario_index]
        cloth_id = dataset[CLOTH_ID_KEY][self.scenario_index]
        # seq_rigid = dataset[RIGID_KEY][self.scenario_index, frame_index, :num_rigid, :]  # (numrigid, 4), xyzr

        seq_rigid = self.dataset_rigid[self.scenario_index, frame_index, :num_rigid, :]
        mesh_tx_list = []
        for obj_i in range(num_rigid):
            # get the origin of each rigid object
            xyz = seq_rigid[obj_i][0:3]
            # get the radius of each rigid object
            r = seq_rigid[obj_i][3]
            # create the sphere in open3d
            mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=r)
            # translate the sphere object according to the origin position
            mesh_tx: open3d.geometry.TriangleMesh = mesh_sphere.translate(xyz)
            #mesh_tx.compute_triangle_normals()
            mesh_tx.compute_vertex_normals()
            mesh_tx.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_tx_list.append(mesh_tx)

        seq = self.dataset_cloth[self.scenario_index, frame_index, :]  # (numpoint, 3)
        # g1 = copy.copy(seq[:, 0])
        # g2 = copy.copy(seq[:, 2])
        # seq[:,2] = g1
        # seq[:,0] = g2
        conn = self.data.topodict[cloth_id]
        cloth_mesh = open3d.geometry.TriangleMesh()
        cloth_mesh.vertices = open3d.utility.Vector3dVector(seq)
        cloth_mesh.triangles = open3d.utility.Vector3iVector(conn)
        cloth_mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((seq.shape[0], 3), 0.5))
        #cloth_mesh.compute_triangle_normals()
        cloth_mesh.compute_vertex_normals()

        if self.show_cloth_mesh:
            mesh_tx_list.append(cloth_mesh)

        #######################
        total_points = seq
        cloth_pcd = open3d.geometry.PointCloud()
        color_point = np.full(total_points.shape, np.array([0.0, 0.0, 0.0]))
        for i in self.keypoint_indices:
            color_point[i] = np.array([0.8, 0.0, 0.0])
        color_point[self.keypoint_index] = np.array([1.0, 0.0, 0.0])

        #if self.show_cloth_mesh:
        #    cloth_pcd.points = open3d.utility.Vector3dVector(total_points)
        #    cloth_pcd.colors = open3d.utility.Vector3dVector(color_point)
        #else:
        cloth_pcd.points = open3d.utility.Vector3dVector(total_points[self.keypoint_indices])
        cloth_pcd.colors = open3d.utility.Vector3dVector(color_point[self.keypoint_indices])
        mesh_tx_list.append(cloth_pcd)

        ##########################
        # effector
        if useeffector == Datasets.EffectorMotion.Ball:
            seq_effector = dataset[EFFECTOR_KEY][self.scenario_index, frame_index, :][0]
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
        if False:
            mesh_tx_list.append(line_set)

        line_mesh = LineMesh(line_set.points, line_set.lines, line_set.colors, radius=0.01)
        for geom in line_mesh.cylinder_segments:
            mesh_tx_list.append(geom)

        return DataVisualizer.Frame(mesh_tx_list, cloth_mesh, cloth_pcd, color_point)

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

    def on_start_video(self, o3d_vis: open3d.visualization.VisualizerWithKeyCallback):
        self.frame_index = 0
        self.playing_video = True

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
        GLFW_KEY_V = 86
        GLFW_KEY_SPACE = 32

        o3d_vis.register_key_callback(GLFW_KEY_ESCAPE, self.on_quit)
        o3d_vis.register_key_callback(GLFW_KEY_LEFT, self.on_prev_frame)
        o3d_vis.register_key_callback(GLFW_KEY_RIGHT, self.on_next_frame)
        o3d_vis.register_key_callback(GLFW_KEY_DOWN, self.on_prev_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_UP, self.on_next_scenario)
        o3d_vis.register_key_callback(GLFW_KEY_A, self.on_prev_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_S, self.on_next_keypoint)
        o3d_vis.register_key_callback(GLFW_KEY_V, self.on_start_video)
        o3d_vis.register_key_callback(GLFW_KEY_SPACE, self.on_toggle_cloth_mesh)

    def run(self):
        o3d_vis = open3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.create_window()
        render_option: open3d.visualization.RenderOption = o3d_vis.get_render_option()
        render_option.point_size = 15
        # Line width is no longer supported by newer OpenGL versions
        render_option.line_width = 10
        render_option.light_on = True
        render_option.mesh_shade_option = open3d.visualization.MeshShadeOption.Color
        render_option.mesh_color_option = open3d.visualization.MeshColorOption.Color
        render_option.mesh_show_back_face = True
        render_option.show_coordinate_frame = True

        self.register_callbacks(o3d_vis)

        # TODO: Keep time difference to advance frames
        last_time = datetime.now()

        old_scenario_index = -1
        old_frame_index = -1
        old_keypoint_index = -1
        old_show_cloth_mesh = self.show_cloth_mesh
        old_playing_video = self.playing_video
        frame_times = []

        max_frame_time = 0.10
        frame_time = max_frame_time

        self.running = True
        while self.running:
            # running, scenario_index and frame_index are modified by the keypress events
            scenario_changed = self.scenario_index != old_scenario_index
            show_cloth_mesh_changed = self.show_cloth_mesh != old_show_cloth_mesh

            if old_playing_video != self.playing_video:
                # Reset the frame advance timer if we start playing video
                last_time = datetime.now()
                old_playing_video = self.playing_video

            current_time = datetime.now()
            time_difference = (current_time - last_time).total_seconds()
            if self.playing_video and time_difference > frame_time:
                self.frame_index += 1
                frame_times.append(time_difference)
                if self.frame_index >= len(self.frames) - 1:
                    # Stop playing video at the last frame
                    self.frame_index = len(self.frames) - 2
                    self.playing_video = False
                    print("Mean frame time:", np.mean(frame_times))
                    frame_times = []
                # If we used more time than the maximal frame time we try to render the next frame faster
                frame_time = max_frame_time - (time_difference - frame_time)
                last_time = current_time

            old_scenario_index = self.scenario_index
            if scenario_changed or show_cloth_mesh_changed:
                self.frames = self.load_frames()

            frame_changed = self.frame_index != old_frame_index
            keypoint_changed = self.keypoint_index != old_keypoint_index

            frame = self.frames[self.frame_index]
            if keypoint_changed:
                was_keypoint = old_keypoint_index in self.keypoint_indices
                #frame.update_keypoint(old_keypoint_index, self.keypoint_index, was_keypoint, o3d_vis)
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



class Evaluation_Visual:
    def __init__(self, model: PredictionInterface,
                 max_scenarios: int = 0, useeffector: int=0):
        self.model = model
        self.max_scenario_index = max_scenarios
        # newdata = copy.copy(data)
        newdata = None

        self.keypoint_indices = keypoint_indices
        self.keypoint_edges = keypoint_edges
        # self.data_vis = KeypointDataVisualizer(newdata, scenario_index, self.keypoint_indices, self.keypoint_edges)
        self.data_vis = None
        self.useeffector = useeffector

    def evaluate(self, data: SimulatedData) -> KeypointDataVisualizer:
        newdata = copy.copy(data)
        self.data_vis = KeypointDataVisualizer(newdata, 0, self.keypoint_indices, self.keypoint_edges, useeffector=self.useeffector)
        self.data_vis.show_cloth_mesh = False

        self.calculate_keypoint_pos(data)
        return self.data_vis

    def calculate_keypoint_pos(self, data: SimulatedData):
        num_scenarios = min(data.num_scenarios, self.max_scenario_index)
        print("Evaluating horizon prediction by visual inspection")
        for scenario_index in tqdm.tqdm(range(num_scenarios)):
            scenario = data.scenario(scenario_index)
            self.model.prepare_scenario(scenario)

            current_frame = scenario.frame(0)
            next_frame = scenario.frame(1)

            next_effector_position = next_frame.get_effector_pose()[0]
            hand_left_xyz_next = next_frame.get_left_hand_position()
            hand_right_xyz_next = next_frame.get_right_hand_position()
            prev_predicted_frame = self.model.predict_frame(current_frame, next_effector_position,
                                                            hand_left_xyz_next, hand_right_xyz_next)

            for frame_index in range(1, data.num_frames - 1):
                current_frame = next_frame
                current_frame.overwrite_keypoint_positions(prev_predicted_frame.cloth_keypoint_positions)
                current_frame.overwrite_rigid_body_positions(prev_predicted_frame.rigid_body_positions)

                next_frame = scenario.frame(frame_index + 1)
                next_effector_position = next_frame.get_effector_pose()[0]
                hand_left_xyz_next = next_frame.get_left_hand_position()
                hand_right_xyz_next = next_frame.get_right_hand_position()

                # Evaluate single frame
                predicted_frame = self.model.predict_frame(current_frame, next_effector_position,
                                                           hand_left_xyz_next, hand_right_xyz_next)

                self.data_vis.dataset_cloth[scenario_index][frame_index][
                    self.keypoint_indices] = predicted_frame.cloth_keypoint_positions

                numrigid = predicted_frame.rigid_body_positions.shape[0]
                self.data_vis.dataset_rigid[scenario_index][frame_index][:numrigid,:3] = predicted_frame.rigid_body_positions

                prev_predicted_frame = predicted_frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the predicting results for deformable bag manipulation')
    parser.add_argument('--model', help='Specify the model name: one-stage, two-stage, horizon',
                        default='one-stage')
    parser.add_argument('--max_scenarios', type=int, default=10)
    parser.add_argument('--set_name', type=str, default="train")
    parser.add_argument('--task_index', type=int, default=1)
    parser.add_argument('--demo', type=bool, default=False)

    args, _ = parser.parse_known_args()

    max_scenarios = args.max_scenarios

    tasks_path = "./h5data/tasks"

    print("Chosen task:", args.task_index)
    task = Datasets.get_task_by_index(args.task_index)
    task.isdemo = args.demo
    useeffector = task.effector_motion

    subset = Datasets.Subset.from_name(args.set_name)
    if subset is None:
        raise ValueError(f"Subset with name '{args.set_name}' is unknown")

    path_to_dataset = task.path_to_dataset(tasks_path, subset)
    path_to_topodict = task.path_to_topodict(tasks_path, subset)

    # Use a separate path to store the models for each task
    models_root_path = f"./models/task-{task.index}/"

    # load the dataset
    dataset = SimulatedData.load(path_to_topodict, path_to_dataset)

    # create prediction model
    model_name = args.model
    model = create_prediction_model(model_name, models_root_path)

    print(args.task_index)
    #
    evaluation_visual = Evaluation_Visual(model, max_scenarios=max_scenarios, useeffector=useeffector)
    data_vis = evaluation_visual.evaluate(dataset)

    data_vis.run()
