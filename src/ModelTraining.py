"""
Training a model given by a specification
"""

import Models
import ModelSpecification
from ModelTrainer import ModelTrainer
import Datasets

import argparse
import os

parser = argparse.ArgumentParser(description='Train a prediction model for deformable bag manipulation')
parser.add_argument('--spec', help='Specify the model specification you want to train: motion, has_moved',
                    default='motion')
parser.add_argument('--frame_step', type=int, default=1)
parser.add_argument('--augment_rotation', type=bool, default=False)
parser.add_argument('--task_index', type=int, default=None)

args, _ = parser.parse_known_args()

# Specification of the model which we want to train
if args.spec == "motion":
    model = Models.specify_motion_model("MotionModel")
elif args.spec == "has_moved":
    model = Models.specify_has_moved_model("HasMovedModel")
else:
    raise NotImplementedError("Model specification is unknown", args.spec)

# Add a suffix for the frame_step to distinguish longer horizon prediction models
model.name = model.name + "_" + str(args.frame_step)

# For frame-wise prediction set frame_step to 1
# For long horizon prediction choose a value > 1
model.training_params.frame_step = args.frame_step

# augmentation for learning rotation invariant
model.training_params.augment_rotation = args.augment_rotation

# TODO: Mirror augmentation

print("Training ", model.name, "with frame_step", model.training_params.frame_step)

if args.task_index is None:
    # Paths to training and validation datasets (+ topology of the deformable object)
    train_path_to_topodict = 'h5data_archive/topo_train.pkl'
    train_path_to_dataset = 'h5data_archive/train_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

    valid_path_to_topodict = 'h5data_archive/topo_valid.pkl'
    valid_path_to_dataset = 'h5data_archive/valid_sphere_sphere_f_f_soft_out_scene1_2TO5.h5'

    models_root_path = "./models/"
else:
    tasks_path = "./h5data/tasks"

    print("Chosen task:", args.task_index)
    task = Datasets.get_task_by_index(args.task_index)
    if task is None:
        print("Could not load task:", args.task_index)
        exit()

    train_path_to_dataset = task.path_to_dataset(tasks_path, Datasets.Subset.Training)
    train_path_to_topodict = task.path_to_topodict(tasks_path, Datasets.Subset.Training)

    valid_path_to_dataset = task.path_to_dataset(tasks_path, Datasets.Subset.Validation)
    valid_path_to_topodict = task.path_to_topodict(tasks_path, Datasets.Subset.Validation)

    # Use a separate path to store the models for each task
    models_root_path = f"./models/task-{task.index}/"

    # Adapt model specs to task (store action parameters in global graph features)
    if task.effector_motion == Datasets.EffectorMotion.Ball:
        model.input_graph_format.global_format = ModelSpecification.GlobalFormat.NextEndEffectorXYZR
    elif task.left_hand_motion in [Datasets.HandMotion.Circle, Datasets.HandMotion.Open, Datasets.HandMotion.Lift]:
        model.input_graph_format.global_format = ModelSpecification.GlobalFormat.NextHandPositionXYZ_Left
    elif task.right_hand_motion in [Datasets.HandMotion.Circle, Datasets.HandMotion.Open, Datasets.HandMotion.Lift]:
        model.input_graph_format.global_format = ModelSpecification.GlobalFormat.NextHandPositionXYZ_Right
    else:
        raise NotImplementedError("Action encoding for task is not specified")


# Ensure that the root path for storing model state and checkpoints exists
if not os.path.exists(models_root_path):
    os.makedirs(models_root_path)


trainer = ModelTrainer(model=model,
                       models_root_path=models_root_path,
                       train_path_to_topodict=train_path_to_topodict,
                       train_path_to_dataset=train_path_to_dataset,
                       valid_path_to_topodict=valid_path_to_topodict,
                       valid_path_to_dataset=valid_path_to_dataset,
                       )

trainer.train()


