"""
Generate diagrams from the evaluation data
"""

import Datasets

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from typing import Tuple, List
import argparse
import csv
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# We only evaluate single time step prediction errors for the one-stage and two-stage models.
SINGLE_TIME_STEP_PREDICTION_MODELS = [
    {"id": "one-stage", "name": "One-Stage"},
    {"id": "two-stage", "name": "Two-Stage"},
]

# For the horizon prediction, we evaluate all three models
LONG_HORIZON_PREDICTION_MODELS = [
    {"id": "one-stage", "name": "One-Stage"},
    {"id": "two-stage", "name": "Two-Stage"},
    {"id": "horizon", "name": "Mixed-Horizon"}
]

def load_error_stats_for_subset(eval_path: str, subset: Datasets.Subset, models: list) -> Tuple[np.array, np.array]:
    # Load errors
    mean_errors = np.zeros(len(models))
    stddevs = np.zeros(len(models))
    for i, model in enumerate(models):
        filename = f"error_{subset.filename()}_{model['id']}.csv"
        full_path = os.path.join(eval_path, filename)
        if not os.path.exists(full_path):
            print("Could not open file:", full_path)
            return None, None

        with open(full_path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            rows = [row for row in reader]
            # The first row contains the column names, so skip it
            values = np.array(rows[1:][0], np.float32)
            mean_errors[i] = values[0]
            stddevs[i] = values[1]

    return mean_errors, stddevs


def load_task_error_stats(eval_path: str):
    # The
    models = SINGLE_TIME_STEP_PREDICTION_MODELS

    errors_per_subset = dict()

    for subset in Datasets.Subset:
        mean_errors, _ = load_error_stats_for_subset(eval_path, subset, models)
        if mean_errors is None:
            return None

        errors_per_subset[subset] = mean_errors

    return errors_per_subset


def load_complete_error_stats():
    models = SINGLE_TIME_STEP_PREDICTION_MODELS

    error_stats = dict()
    for subset in Datasets.Subset:
        error_stats[subset] = []
        for i, model in enumerate(models):
            error_stats[subset].append([])

    for task in Datasets.tasks:
        # Use a separate path to store the models for each task
        eval_path = f"./models/task-{task.index}/evaluation"

        if not os.path.exists(eval_path):
            print(f"No evaluation directory found for task {task.index}: {eval_path}")
            continue

        errors_per_subset = load_task_error_stats(eval_path)
        if errors_per_subset is None:
            continue

        # FIXME: There must be a better way to construct this data structure
        for subset in Datasets.Subset:
            for i, model in enumerate(models):
                error = errors_per_subset[subset][i]
                error_stats[subset][i].append(error)

    return error_stats


def save_error_plot(eval_path: str, filename: str):
    fig, ax = plt.subplots()
    ax.set_ylabel('Mean Position Error')

    model_names = [model['name'] for model in models]
    x_pos = np.arange(len(model_names))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title(f"Single Frame Prediction: Mean Position Error")

    for i, set in enumerate(Datasets.Subset):
        mean_errors, stddevs = load_error_stats_for_subset(eval_path, set, models)
        if mean_errors is None:
            continue

        pos = x_pos + (i-1) * 0.25

        yerr = stddevs if plot_stddev_whiskers else None
        ax.bar(pos, mean_errors, width=0.25, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)

    plt.close(fig)


def load_horizon_stats(eval_path: str, subset: Datasets.Subset, models: list) -> np.array:
    # Load errors
    errors = [None] * len(models)
    for i, model in enumerate(models):
        filename = f"horizon_{subset.filename()}_{model['id']}.csv"
        full_path = os.path.join(eval_path, filename)
        if not os.path.exists(full_path):
            print("Could not open file:", full_path)
            return None

        with open(full_path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            rows = [row[0] for row in reader]
            # The first row contains the column names
            # We set it to zero (0-frame prediction has 0 error)
            rows[0] = 0.0
            errors[i] = rows

    return np.array(errors, np.float32)


def save_horizon_plot(eval_path: str, subset: Datasets.Subset, filename: str):
    errors = load_horizon_stats(eval_path, subset, models)
    if errors is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylabel('Mean Position Error')

    x_pos = np.arange(errors.shape[1])

    model_names = [model['name'] for model in models]

    ax.set_xticks(x_pos)
    ax.set_title(f"Horizon Prediction: Mean Position Error ({subset.name})")

    for i, frame_errors in enumerate(errors):
        ax.plot(x_pos, frame_errors, label=model_names[i])

    ax.legend()
    ax.set_ylim(0)
    ax.set_xlim(0, x_pos[-1])

    plt.tight_layout()
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)

    plt.close(fig)


def save_error_bar_plot(error_stats, eval_path: str, filename: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel('Mean Position Error')

    models = SINGLE_TIME_STEP_PREDICTION_MODELS
    model_names = [model['name'] for model in models]
    x_pos = np.arange(len(model_names))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title(f"Single Time Step Prediction Error")

    for i, subset in enumerate(Datasets.Subset):
        errors_in_subset = error_stats[subset]
        # Calculate the mean prediction error (and standard deviation) over all tasks
        mean_errors = [np.mean(task_errors) for task_errors in errors_in_subset]
        stddevs = [np.std(task_errors) for task_errors in errors_in_subset]

        pos = x_pos + (i - 1) * 0.25

        yerr = stddevs if plot_stddev_whiskers else None
        ax.bar(pos, mean_errors, width=0.25, yerr=yerr, label=subset.name,
               align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.legend()
    ax.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)

    plt.close(fig)


def group_tasks_per_action():
    # Group tasks by action
    tasks_per_action = dict()
    for task in Datasets.tasks:
        action = task.action()
        tasks_per_action.setdefault(action, []).append(task)
    return tasks_per_action


def load_long_horizon_stats_for_task(task: Datasets.TaskDataset):
    task_eval_path = f"./models/task-{task.index}/evaluation"
    errors = load_horizon_stats(task_eval_path, Datasets.Subset.Test, LONG_HORIZON_PREDICTION_MODELS)
    return errors


def load_long_horizon_stats_for_tasks(tasks: List[Datasets.TaskDataset]):
    stats_per_task = []
    for task in tasks:
        task_stats = load_long_horizon_stats_for_task(task)
        if task_stats is not None:
            stats_per_task.append(task_stats)
    return stats_per_task


def load_long_horizon_stats():
    # Group tasks by action
    tasks_per_action = group_tasks_per_action()
    stats_per_action = dict()
    for action, tasks in tasks_per_action.items():
        stats_per_action[action] = load_long_horizon_stats_for_tasks(tasks)
    return stats_per_action


def save_long_horizon_plot(eval_path: str, action: Datasets.Action, stats: List[np.array]):
    errors = np.mean(stats, axis=0)
    stddevs = np.std(stats, axis=0)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_ylabel('Mean Position Error')
    ax.set_xlabel('Time Step')

    x_pos = np.arange(errors.shape[1])

    model_names = [model['name'] for model in LONG_HORIZON_PREDICTION_MODELS]

    ax.set_xticks(x_pos[0::5])
    ax.set_title(f"Action: {action.plot_name()}")

    for i, frame_errors in enumerate(errors):
        ax.plot(x_pos, frame_errors, label=model_names[i])
        stddev = stddevs[i]
        ax.fill_between(x_pos, frame_errors - stddev, frame_errors + stddev, alpha=0.2)

    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.45)
    ax.set_xlim(0, x_pos[-1])

    plt.tight_layout()
    filename = f"evaluation_long_horizon_{action.name}.png"
    path = os.path.join(eval_path, filename)
    plt.savefig(path)
    print("Saved:", path)

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a prediction model for deformable bag manipulation')
    parser.add_argument('--plot_stddev_whiskers', type=bool, default=True)

    args, _ = parser.parse_known_args()

    plot_stddev_whiskers = args.plot_stddev_whiskers#

    eval_path = "./evaluation"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    error_stats = load_complete_error_stats()
    save_error_bar_plot(error_stats, eval_path, "evaluation_error_bar_plot.png")

    long_horizon_stats = load_long_horizon_stats()
    for action, stats in long_horizon_stats.items():
        save_long_horizon_plot(eval_path, action, stats)

    if False:
        models = [
            {"id": "one-stage", "name": "One-step"},
            {"id": "two-stage", "name": "Two-step"},
            {"id": "horizon", "name": "Horizon"}
        ]

        for task in Datasets.tasks:
            print("Creating evaluation diagrams for task:", task.index)

            # Use a separate path to store the models for each task
            eval_path = f"./models/task-{task.index}/evaluation"

            if not os.path.exists(eval_path):
                print(f"No evaluation directory found for task {task.index}: {eval_path}")
                continue

            plot_filename = "plot_error_bars.png"
            save_error_plot(eval_path, plot_filename)

            for subset in Datasets.Subset:
                plot_filename = f"plot_horizon_bars_{subset.filename()}.png"
                save_horizon_plot(eval_path, subset, plot_filename)





