# 🚓 Abnormal Driving Detection System  
AI 기반 차량 및 운전자 이상행동 감지 시스템

> 🏆 KT와 함께하는 AI 경진대회 (2024) - **Top 3 수상작**  
> 팀명: **휴지공조**

---

## 📌 Overview

본 프로젝트는 음주운전, 졸음운전, 급발진 등 이상 차량의 움직임을 감지하고 실시간으로 대응 가능한 지능형 시스템을 구축하는 것을 목표로 합니다.  
CCTV 영상을 기반으로 정상 차량의 주행 패턴을 학습하고, 이상 움직임을 자동으로 탐지하여 신고까지 연계할 수 있는 구조를 갖추고 있습니다.

---

## 🧠 Key Features

- **차선 및 차량 객체 인식** (YOLOv3 + OpenCV)
- **배경 제거 및 노이즈 처리** (MOG + Morphology)
- **Optical Flow 기반 주행 방향 및 이상 탐지**
- **정상 패턴 학습을 통한 이상 탐지 모델 구성**
- **자동 신고 시스템 연계 시나리오 제시**

---

## 🗂 Project Structure

| 파일명 | 설명 |
|--------|------|
| `lane_detection.py` | 영상에서 흰색 차선을 인식하여 윤곽선을 시각화합니다. (OpenCV 기반) |
| `car_detection_yolo.py` | YOLOv3를 활용하여 차량을 검출하고 클래스 라벨과 함께 표시합니다. |
| `combined_lane_car_detection.py` | 차선 인식과 차량 인식을 동시에 수행하여 통합적인 도로 인식 환경을 구성합니다. |
| `abnormal_behavior_detection.py` | Optical Flow 기반으로 차량 이동 방향, 거리, 각도를 분석하여 이상 행동을 판단합니다. 이상각도에 대해 경고 시각화도 함께 수행됩니다. |

---

## 🛠 기술 스택

- `Python`, `OpenCV`, `YOLOv3 (Darknet)`, `Numpy`, `Matplotlib`
- `sklearn`, `LinearRegression`, `MOG`, `Optical Flow`
- 📹 영상 처리 및 시각화 중심의 구현

---

## 📷 Demo

| 차선/차량 인식 | Optical Flow 기반 이상 탐지 |
|----------------|-----------------------------|
|![image](https://github.com/user-attachments/assets/892cc65a-f061-4244-bbb7-c87a69fc4d41) | ![image](https://github.com/user-attachments/assets/25b5ad13-8708-41a8-95fb-88aad2846c43) |

> 전체 시연 영상은 `results/` 폴더 참고

---

## 💡 향후 발전 방향

- 음주 감지 센서와 연동한 실시간 판단 시스템
- 신고 기준 강화 및 자동 판단 모델 개선
- GPS 기반 실시간 위치 연동 및 인근 경찰서 자동 알림 기능 추가

