import freenect
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from PIL import Image
from joblib import load
from skimage.color import rgb2gray
from feature_extraction import extract_features

cv2.namedWindow('Cropped Video')
cv2.namedWindow('Cropped Mask')
cv2.namedWindow('Video')
model = load('libras.joblib') 
print('Press ESC in window to stop')

def get_depth():
    depth = freenect.sync_get_depth()[0]
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def get_video():
    video = freenect.sync_get_video()[0]
    return video[:, :, ::-1]  # RGB -> BGR

def get_hand_box(rgb_frame):
    d_frame = freenect.sync_get_depth()[0]
    mask = d_frame<min(d_frame.flatten())+70
    for i, line in enumerate(mask):
        mask[i] = shift(line, -35, cval=False)
    conn = cv2.connectedComponentsWithStats(np.array(mask, dtype=np.uint8)*255, 8, cv2.CV_32S)
    
    num_labels = conn[0]
    labels = conn[1]
    stats = conn[2] # containing total area; hand expected to be in index 1 (second largest area after background) 
    centroids = conn[3]
    if (len(stats)>1 and stats[1][4] > 5000):    
        rgb_frame = np.array(rgb_frame)
        x1, y1 = (int(centroids[1][0])-110, int(centroids[1][1])-145)
        x2, y2 = (int(centroids[1][0])+110, int(centroids[1][1])+85)
        
        cropped_image = video[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        cropped_mask = mask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        return stats[1][4], centroids[1], [x1, y1, x2, y2], cropped_image, cropped_mask
    return -1, -1, [], [], []

while True:
    video = get_video()
    depth = get_depth()
    area, centroid, rect, cropped_video, cropped_mask = get_hand_box(video)
    if (area>0):
        [x1, y1, x2, y2] = rect
        cropped_mask = np.array(cropped_mask, dtype=np.uint8)*255
        video = cv2.rectangle(np.array(video), (x1, y1), (x2, y2), (0,0,255), 2)
        
        cv2.imshow('Cropped Video', cropped_video)
        cv2.imshow('Cropped Mask', cropped_mask)

        cropped_video = cv2.resize(cropped_video, (50,50), interpolation = cv2.INTER_AREA)
        cropped_video = rgb2gray(cropped_video)
        cropped_mask = cv2.resize(cropped_mask, (50,50), interpolation = cv2.INTER_AREA)
        X = extract_features(cropped_video, cropped_mask)
        print(model.predict([X]))
    cv2.imshow('Video', video)
    if cv2.waitKey(10) == 27:
        break
