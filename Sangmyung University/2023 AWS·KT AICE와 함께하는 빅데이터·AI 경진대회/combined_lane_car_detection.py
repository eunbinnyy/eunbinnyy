import cv2
import numpy as np
import pandas as pd
import openpyxl
from sklearn.linear_model import LinearRegression

video_file = "cctv_result_mplg.avi"
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

frame_width, frame_height, frame_rate = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("optical_result.avi", fourcc, frame_rate, (frame_width, frame_height), 1)

# 엑셀 파일이 존재하지 않을 경우 새로운 워크북 생성
try:
    sheet = openpyxl.load_workbook('optical_flow_results.xlsx')
except FileNotFoundError:
    sheet = openpyxl.Workbook()
    redf = sheet.active
    redf.append(['frame_num', 'x', 'y', 'dx', 'dy', 'dist', 'angle'])
else:
    redf = sheet.active

def drawFlow(img, flow, step=20):
    h, w = img.shape[:2]
    idx_y, idx_x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)
    
    for x, y in indices:
        dx, dy = flow[y, x].astype(int)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.rad2deg(np.arctan(-dy/dx))
        if angle < 0:
            angle = 360 + angle
        if dist != 0:
            print(f" ({x}, {y})", dx, dy, dist, angle)
            cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)

prev = None
count = 1

# indices 변수를 정의
h, w = frame_height, frame_width
step = 20
max_x = w - step
max_y = h - step
idx_y, idx_x = np.mgrid[step // 2:max_y:step, step // 2:max_x:step].astype(int)
indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

# 회귀 모델을 저장할 딕셔너리 생성
models = {}

# 이상치를 판단하기 위한 threshold 값 설정
threshold = 10  # 예시 값, 필요에 따라 조정

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
        flow = cv2.calcOpticalFlowFarneback(prev, gray, flow=None, pyr_scale=0.7, levels=3, winsize=50, iterations=3, poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        for y, x in indices:
            if 0 <= y < h and 0 <= x < w:
                dx, dy = flow[y, x].astype(int)
                dist = np.sqrt(dx ** 2 + dy ** 2)
                angle = np.rad2deg(np.arctan2(dy, dx))
                if angle < 0:
                    angle = 360 + angle
                if dist != 0:
                    print(f" ({x}, {y})", dx, dy, dist, angle)
                    cv2.line(frame, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)
                    new = [count, x, y, dx, dy, dist, angle]
                    redf.append(new)
                    
                    # (x, y)별 회귀 모델 생성 및 저장
                    key = (x, y)
                    if key not in models:
                        models[key] = LinearRegression()
                    X = np.array([[dx, dy]])
                    y = np.array([angle])
                    models[key].fit(X, y)
                    
                    # dx, dy를 모델에 입력하여 예측된 angle 값과 비교하여 이상치 판단
                    predicted_angle = models[key].predict(X)
                    angle_diff = np.abs(angle - predicted_angle)
                    if angle_diff > threshold:  # threshold 값 설정 필요
                        print(f"Anomaly detected at ({x}, {y})! Angle diff: {angle_diff}")

        prev = gray

    cv2.imshow('OpticalFlow', frame)
    out.write(frame)
    count += 1
    if (cv2.waitKey(delay) & 0xFF) == ord('q'):
        break

# 워크북 저장
output_excel_file = "optical_flow_results.xlsx"
sheet.save(output_excel_file)
sheet.close()

cap.release()
out.release()
cv2.destroyAllWindows()


print(models)

'''
# 딕셔너리의 내용을 데이터프레임으로 변환
model_df = pd.DataFrame(list(models.items()), columns=['x_y', 'model'])
# 데이터프레임을 엑셀 파일로 저장
model_excel_file = "models.xlsx"
model_df.to_excel(model_excel_file, index=False)
'''
