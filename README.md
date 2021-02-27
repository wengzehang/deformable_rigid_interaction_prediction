# Graph-based Task-specific Prediction Models for Interactions between Deformable and Rigid Objects

This repository contains the code for learning to predict action effects in scenes with rich interactions between a deformable bag and multiple rigid objects.
For this purpose, we generated a [dataset](https://github.com/wengzehang/deformable_rigid_interaction_prediction/blob/main/docs/dataset.md) containing actions like pushing an object towards the bag, opening the bag, lifting the bag and moving the handles of the bag along a trajectory.

## Prediction Models
We propose a Position Prediction Module (PPM) and an Active Prediction Module (APM), both based on [graph nets](https://github.com/deepmind/graph_nets).
We compare a one-stage model (PPM alone) with a two-stage model (APM + PPM) and show the benefits of the two-stage approach.
