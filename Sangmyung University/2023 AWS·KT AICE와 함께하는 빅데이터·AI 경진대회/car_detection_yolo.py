import cv2
import numpy as np
import matplotlib.pyplot as plt

# 비디오 파일 이름 설정
video_file = "cctv_result.avi"
# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_file)
# 비디오 프레임 속도(fps) 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
# 각 프레임 간 지연 시간 계산
delay = int(1000 / fps)
# 비디오 프레임 너비, 높이, 프레임 속도 가져오기
frame_width, frame_height, frame_rate = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
# VideoWriter에 사용할 fourcc 코드 생성 (DIVX 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# VideoWriter 객체 생성
out = cv2.VideoWriter("cctv_result_mplg.avi", fourcc, frame_rate, (frame_width, frame_height), 0)

# BackgroundSubtractorMOG 배경 제거 모델 생성
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# 형태학적 연산을 위한 커널 생성 (타원 형태)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 비디오 프레임을 읽어오며 처리하는 루프
while cap.isOpened():
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 배경 제거 모델을 사용하여 전경 마스크 생성
    fgmask = fgbg.apply(frame)
    
    # 형태학적 열기 연산을 통해 전경 마스크의 노이즈 제거
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # 전경 마스크를 화면에 표시
    cv2.imshow('MOG', fgmask)
    
    # 전경 마스크를 출력 비디오로 저장
    out.write(fgmask)

    # 'q' 키를 누를 경우 루프 종료
    if (cv2.waitKey(delay) & 0xFF) == ord('q'):
        break
    
# 캡처 객체와 출력 객체 해제
cap.release()
out.release()

# 모든 창 닫기
cv2.destroyAllWindows()
