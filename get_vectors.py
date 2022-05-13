import cv2
import numpy as np

from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator

face_detector = FaceDetector('models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
facial_landmarks_detector = FacialLandmarksDetector('models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
head_pose_estimator = HeadPoseEstimator('models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
gaze_estimator = GazeEstimator('models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
face_detector.load_model()
facial_landmarks_detector.load_model()
head_pose_estimator.load_model()
gaze_estimator.load_model()

def get_crop_image(image, box):
    xmin, ymin, xmax, ymax = box
    crop_image = image[ymin:ymax, xmin:xmax]
    return crop_image

def get_vectors(frame):
    face_box = face_detector.predict(frame)[0]
    face_image = get_crop_image(frame, face_box)
    eye_boxes, eye_centers = facial_landmarks_detector.predict(face_image)
    left_eye_image, right_eye_image = [get_crop_image(face_image, eye_box) for eye_box in eye_boxes]
    head_pose_angles = head_pose_estimator.predict(face_image)
    return gaze_estimator.predict(right_eye_image, head_pose_angles, left_eye_image)    
    
