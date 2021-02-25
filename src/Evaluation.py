"""
Evaluation code for prediction models
"""

from NewPredictionModels import *
from SimulatedData import SimulatedData, Frame, keypoint_indices
import Datasets

import numpy as np
import argparse
import tqdm
import csv
import os


class EvaluationResult:
    def __init__(self,
                 cloth_pos_error_mean: float,
                 cloth_pos_error_stddev: float,
                 rigid_pos_error_mean: float,
                 rigid_pos_error_stddev: float,
                 horizon_pos_error_mean: np.array):
        self.cloth_pos_error_mean = cloth_pos_error_mean
        self.cloth_pos_error_stddev = cloth_pos_error_stddev
        self.rigid_pos_error_mean = rigid_pos_error_mean
        self.rigid_pos_error_stddev = rigid_pos_error_stddev
        self.horizon_pos_error_mean = horizon_pos_error_mean


class Evaluation:
    def __init__(self, model: PredictionInterface,
                 max_scenario_index: int = 100):
        self.model = model
        self.max_scenario_index = max_scenario_index

    def evaluate_dataset(self, data: SimulatedData) -> EvaluationResult:

        cloth_pos_error_mean, cloth_pos_error_stddev, rigid_pos_error_mean, rigid_pos_error_stddev = \
            self.calculate_keypoint_pos_mean_error(data)
        horizon_pos_error_mean = self.calculate_horizon_pos_error(data)

        return EvaluationResult(cloth_pos_error_mean, cloth_pos_error_stddev,
                                rigid_pos_error_mean, rigid_pos_error_stddev,
                                horizon_pos_error_mean)

    def calculate_keypoint_pos_mean_error(self, data: SimulatedData):
        num_scenarios = min(data.num_scenarios, self.max_scenario_index)
        cloth_errors = np.zeros(num_scenarios * (data.num_frames - 1))
        rigid_errors = np.zeros(num_scenarios * (data.num_frames - 1))
        error_index = 0
        print("Evaluating statistics about position error")
        for scenario_index in tqdm.tqdm(range(num_scenarios)):
            scenario = data.scenario(scenario_index)
            self.model.prepare_scenario(scenario)

            next_frame = scenario.frame(0)
            for frame_index in range(data.num_frames - 1):
                current_frame = next_frame
                next_frame = scenario.frame(frame_index + 1)

                # Evaluate single frame
                next_effector_position = next_frame.get_effector_pose()[0]
                hand_left_xyz_next = next_frame.get_left_hand_position()
                hand_right_xyz_next = next_frame.get_right_hand_position()
                predicted_frame = self.model.predict_frame(current_frame, next_effector_position,
                                                           hand_left_xyz_next, hand_right_xyz_next)
                cloth_errors[error_index] = self.calculate_cloth_pos_error(predicted_frame, next_frame)
                rigid_errors[error_index] = self.calculate_rigid_pos_error(predicted_frame, next_frame)

                error_index += 1

        return np.mean(cloth_errors), np.std(cloth_errors), np.mean(rigid_errors), np.std(rigid_errors)

    def calculate_horizon_pos_error(self, data: SimulatedData):
        num_scenarios = min(data.num_scenarios, self.max_scenario_index)
        errors_per_step = np.zeros((data.num_frames - 1, num_scenarios))

        print("Evaluating horizon position error")
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
            errors_per_step[0][scenario_index] = self.calculate_cloth_pos_error(prev_predicted_frame, next_frame)

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
                errors_per_step[frame_index][scenario_index] = \
                    self.calculate_cloth_pos_error(predicted_frame, next_frame)

                prev_predicted_frame = predicted_frame

        mean_error_per_step = np.mean(errors_per_step, axis=-1)
        return mean_error_per_step

    def calculate_cloth_pos_error(self, predicted: PredictedFrame, ground_truth: Frame):
        pred_keypoint_pos = predicted.cloth_keypoint_positions
        gt_keypoint_pos = ground_truth.get_cloth_keypoint_positions(keypoint_indices)

        errors = np.linalg.norm(gt_keypoint_pos - pred_keypoint_pos, axis=-1)
        mean_error = np.mean(errors)
        return mean_error

    def calculate_rigid_pos_error(self, predicted: PredictedFrame, ground_truth: Frame):
        pred_keypoint_pos = predicted.rigid_body_positions
        gt_keypoint_pos = ground_truth.get_rigid_keypoint_info()[:, :3]

        errors = np.linalg.norm(gt_keypoint_pos - pred_keypoint_pos, axis=-1)
        mean_error = np.mean(errors)
        return mean_error


def create_prediction_model(model_name: str, models_root_path: str):
    motion_model_1_spec = ModelSpecification.ModelSpecification(name="MotionModel_1")
    motion_model_1 = MotionModelFromSpecification(motion_model_1_spec, models_root_path)
    if model_name == "one-stage":
        print("Chosen model one-stage:", model_name)
        return motion_model_1

    has_moved_model_1_spec = ModelSpecification.ModelSpecification(name="HasMovedModel_1")
    mask_model_1 = HasMovedMaskModelFromSpecification(motion_model_1,
                                                      has_moved_model_1_spec,
                                                      models_root_path)
    if model_name == "two-stage":
        print("Chosen model two-stage:", model_name)
        return mask_model_1

    motion_model_5_spec = ModelSpecification.ModelSpecification(name="MotionModel_5")
    motion_model_5 = MotionModelFromSpecification(motion_model_5_spec, models_root_path)
    has_moved_model_5_spec = ModelSpecification.ModelSpecification(name="HasMovedModel_5")
    mask_model_5 = HasMovedMaskModelFromSpecification(motion_model_5,
                                                      has_moved_model_5_spec,
                                                      models_root_path)

    horizon_model = HorizonModel(mask_model_1, mask_model_5,
                                 start_horizon_frame=1)

    if model_name == "horizon":
        print("Chosen model horizon:", model_name)
        return horizon_model

    raise NotImplementedError("Model name was not handled in create_prediction_model()", model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a prediction model for deformable bag manipulation')
    parser.add_argument('--model', help='Specify the model name: one-stage, two-stage, horizon',
                        default='one-stage')
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--set_name', type=str, default=None)
    parser.add_argument('--task_index', type=int, default=1)

    args, _ = parser.parse_known_args()

    if args.set_name is None:
        subsets = [s.filename() for s in Datasets.Subset]
    else:
        subsets = [args.set_name]

    for set_name in subsets:
        if args.task_index is None:
            # Paths to training and validation datasets (+ topology of the deformable object)
            path_to_topodict = f'h5data_archive/topo_{set_name}.pkl'
            path_to_dataset = f'h5data_archive/{set_name}_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

            models_root_path = "./models/"
        else:
            tasks_path = "./h5data/tasks"

            print("Chosen task:", args.task_index)
            task = Datasets.get_task_by_index(args.task_index)

            subset = Datasets.Subset.from_name(set_name)
            if subset is None:
                raise ValueError(f"Subset with name '{set_name}' is unknown")

            path_to_dataset = task.path_to_dataset(tasks_path, subset)
            path_to_topodict = task.path_to_topodict(tasks_path, subset)

            # Use a separate path to store the models for each task
            models_root_path = f"./models/task-{task.index}/"

        dataset = SimulatedData.load(path_to_topodict, path_to_dataset)

        max_scenarios = args.max_scenarios
        if max_scenarios is None:
            max_scenarios = dataset.num_scenarios

        model_name = args.model
        model = create_prediction_model(model_name, models_root_path)

        evaluation = Evaluation(model, max_scenario_index=max_scenarios)

        result = evaluation.evaluate_dataset(dataset)

        evaluation_path = os.path.join(models_root_path, "evaluation")
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)

        filename = f"error_{set_name}_{model_name}.csv"
        path = os.path.join(evaluation_path, filename)
        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["cloth_pos_error_mean", "cloth_pos_error_stddev",
                             "rigid_pos_error_mean", "rigid_pos_error_stddev",])
            writer.writerow([result.cloth_pos_error_mean, result.cloth_pos_error_stddev,
                             result.rigid_pos_error_mean, result.rigid_pos_error_stddev])

        filename = f"horizon_{set_name}_{model_name}.csv"
        path = os.path.join(evaluation_path, filename)
        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["pos_error_mean"])
            row_count = result.horizon_pos_error_mean.shape[0]
            for i in range(row_count):
                writer.writerow([result.horizon_pos_error_mean[i]])
