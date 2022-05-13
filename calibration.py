import cv2
import time
import numpy as np
import json
import picamera 
from get_vectors import get_vectors

# addr = 'http://192.168.1.59:8080/video' FOR TESTING RESULT USED IP CAMERA

back_source = cv2.VideoCapture(0)
front_source = cv2.VideoCapture(1)

calibration_stage = 0
calibration_circle_coordinates = {0: (50, 50), 1: (960 - 50, 540 - 50), 2: (960 - 50, 50), 3: (50, 540 - 50), 4: (480, 285)}
calibration_result_vectors = {}
vectors_captured = False

def get_vector_plane_intersection_point(vector):
    x, y, z = vector
    multiplier = 1 / x
    return np.array([y * multiplier, z * multiplier])

def get_current_eye_vectors():
    ret, frame = back_source.read()
    v = tuple(get_vectors(frame))
    print(v)
    return v
    
def add_circle_at_point(frame, point):    
    return cv2.circle(frame, tuple([int(x) for x in point]), 2, (255,0,255), 4)
 
def add_calibration_circle(frame):
    return add_circle_at_point(frame, calibration_circle_coordinates[calibration_stage])        
    
def get_ratios(p_infer, p1, p2):
    p_y, p_z = p_infer
    p1_y, p1_z = p1
    p2_y, p2_z = p2
    yr = (p2_y - p_y) / (p2_y - p1_y)
    zr = (p2_z - p_z) / (p2_z - p1_z)
    return yr, zr
    
def calibrate(vectors):     
    vector_a, vector_b = vectors
    p1 = get_vector_plane_intersection_point(vector_a)
    p2 = get_vector_plane_intersection_point(vector_b)
    #json.dump({'p1': p1, 'p2': p2}, open('calibration.out', 'w'))
    return np.array(p1), np.array(p2)
    return p1, p2

while not vectors_captured:
    ret, frame = back_source.read()
    frame = cv2.resize(frame, (960, 540))
    if calibration_stage <= 4:
        frame = add_calibration_circle(frame)
    cv2.imshow('calibrate', frame)        
    if cv2.waitKey(1) == ord('s') and calibration_stage <= 2:
        calibration_result_vectors[calibration_stage] = get_current_eye_vectors()
        calibration_stage += 1
    if calibration_stage == 2:
        vectors_captured = True
cv2.destroyAllWindows()

p1, p2 = calibrate([calibration_result_vectors[0], calibration_result_vectors[1]])

while True:
    ret, back_frame = back_source.read()
    ret, front_frame = front_source.read()        
    front_frame = cv2.resize(front_frame, (960, 540))
    front_frame = front_frame[50:-50, 50:-50]        
    try:        
        current_vector = get_vectors(back_frame)
        current_point = get_vector_plane_intersection_point(current_vector)
        ratio_x, ratio_y = get_ratios(current_point, p1, p2)
        draw_points = front_frame.shape[0] * ratio_x, front_frame.shape[1] * ratio_y
        print(draw_points)
        front_frame = add_circle_at_point(front_frame, draw_points)
        cv2.imshow('prediction', front_frame)
        cv2.waitKey(1)
    except:
        pass
    
   