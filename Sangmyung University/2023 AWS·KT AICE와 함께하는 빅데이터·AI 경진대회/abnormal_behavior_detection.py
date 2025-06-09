import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#video_file = "cctv_result.avi"
video_file = "cctv_result_mplg.avi"
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

frame_width, frame_height, frame_rate = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter("optical_result.avi", fourcc, frame_rate,(frame_width,frame_height),1)

avg=  140

missing_angles= [1, 2, 3, 4, 13, 19, 20, 22, 24, 25, 27, 31, 42, 43, 44, 46, 47, 88, 89, 100, 103, 105, 106, 109, 115, 117, 118, 119, 120, 121, 122, 124, 125, 127, 129, 131, 132, 133, 134, 136, 137, 139, 142, 145, 147, 148, 151, 152, 155, 156, 157, 159, 162, 163, 166, 167, 169, 173, 174, 175, 176, 177, 178, 179, 182, 183, 185, 186, 193, 196, 197, 202, 205, 207, 211, 223, 224, 226, 238, 242, 244, 250, 252, 256, 266, 267, 268, 269, 271, 272, 283, 287, 289, 292, 295, 297, 298, 301, 312, 313, 314, 316, 317, 322, 328, 331, 332, 334, 335, 337, 340, 342, 343, 346, 349, 355, 356, 357, 358, 359, 360]


def drawFlow(img, flow, step=15):
    h, w = img.shape[:2]
    idx_y, idx_x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)
    
    for x, y in indices:
        dx, dy = flow[y, x].astype(int)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.rad2deg(np.arctan(-dy / dx))
        if angle < 0:
            angle = 360 + angle
        if dist != 0:
            if angle.round() in missing_angles:
                cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print(f"({x}, {y})", dx, dy, dist, angle)
                cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)

           
            

prev= None
count = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev is None:
        prev = gray
    
    else:
        print(f"\n\n--------------------{count}th frame--------------")
        print(" (x,y) dx dy     flowsize     flowdirection")
        flow = cv2.calcOpticalFlowFarneback(prev, gray, flow=None,pyr_scale=0.7,levels=3,winsize=50,iterations=3, poly_n=5,poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        drawFlow(frame, flow)
        prev = gray
        
    cv2.imshow('OpticalFlow', frame) 
    #out.write(frame)
    count += 1
    if (cv2.waitKey(delay) & 0xFF) == ord('q'):
        break
cap.release()
#out.release()
cv2.destroyAllWindows()
