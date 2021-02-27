# deformable_rigid_interaction_prediction
Prediction of deformable rigid object interaction using graph neural network

# Description
## Task id
xxx
## Data type and Network Structure
xxx
## Data Visualization
python visualization/DataVisualizer.py --task_index 3

## train MPM
python ModelTraining.py --spec=motion --frame_step=1 --task_index=1

## train APM
python ModelTraining.py --spec=has_moved --frame_step=1 --task_index=1

## Quantitative evaluation
python Evaluation.py --model=one-stage --task_index=1

## Qualitative evaluation
python visualization/ValidVisualizer.py --model two-stage --set_name test --task_index 1
