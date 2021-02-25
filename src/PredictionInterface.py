"""
Interface for prediction models.
"""

import SimulatedData

import numpy as np
from typing import List


class PredictedFrame:
    def __init__(self,
                 cloth_keypoint_positions: np.array,
                 rigid_body_positions: np.array):
        self.cloth_keypoint_positions = cloth_keypoint_positions
        self.rigid_body_positions = rigid_body_positions


class PredictionInterface:

    def predict_frame(self, frame: SimulatedData.Frame,
                      effector_xyzr_next: np.array,
                      hand_left_xyz_next: np.array,
                      hand_right_xyz_next: np.array) -> PredictedFrame:
        raise NotImplementedError()

    def prepare_scenario(self, scenario: SimulatedData.Scenario):
        # This method is used to predict anchor frames with a long horizon model
        # It is called before any predict_frame() call is made for that scenario
        pass
