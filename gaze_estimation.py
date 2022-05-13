import sys
import logging as log
import cv2
import numpy as np
import math
import time

from inference import Network

class GazeEstimator:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.network = Network(model_name, device, extensions)

    def load_model(self):
        self.network.load_model()

    def predict(self, right_eye_image, head_pose_angles, left_eye_image):
        _, _, roll = head_pose_angles
        right_eye_image, head_pose_angles, left_eye_image, preprocess_input_time = self._preprocess_input(right_eye_image, head_pose_angles, left_eye_image)
        input_dict = {"left_eye_image": left_eye_image, "right_eye_image": right_eye_image, "head_pose_angles": head_pose_angles}
        self.network.exec_net(0, input_dict)
        status = self.network.wait(0)
        if status == 0:
            outputs = self._preprocess_output(self.network.get_output(0))                        
            return outputs

    def _preprocess_input(self, right_eye_image, head_pose_angles, left_eye_image):
        start_preprocess_time = time.time()
        left_eye_image = self._preprocess_eye_image(left_eye_image)
        right_eye_image = self._preprocess_eye_image(right_eye_image)
        head_pose_angles = self._preprocess_angels(head_pose_angles)
        total_preprocess_time = time.time() - start_preprocess_time
        return right_eye_image, head_pose_angles, left_eye_image, total_preprocess_time    

    def _preprocess_angels(self, head_pose_angles):
        input_shape = self.network.get_input_shape("head_pose_angles")
        head_pose_angles = np.reshape(head_pose_angles, input_shape)
        return head_pose_angles

    def _preprocess_eye_image(self, image):
        n, c, h, w = self.network.get_input_shape("left_eye_image")
        input_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))
        return input_image

    def _preprocess_output(self, outputs): 
        gaze_vector = outputs[0]
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        x, y, z = tuple([round(x, 6) for x in gaze_vector_n])
        z = -1 * z
        return z, x, y
