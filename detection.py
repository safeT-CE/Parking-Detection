import cv2
from ultralytics import YOLO
import datetime
import os

model = YOLO("C:/Parking Detection/best.pt")

cap = cv2.VideoCapture("C:/Parking Detection/test.mp4")
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

frame_count = 0
capture_interval = 100
evidence_dir = 'evidence_photos'
if not os.path.exists(evidence_dir):
    os.makedirs(evidence_dir)

violations = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽는 도중 오류가 발생했습니다.")
        break

    # 객체 탐지
    results = model(frame)

    # 탐지 결과 처리
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            if score >= 0.5:
                if cls == 0:  # 점자 블록 클래스 ID
                    braille_block_label = f'점자 블록 {score:.2f}'
                    braille_block_violation = "점자 블록 위반"
                elif cls == 1:  # 횡단보도 클래스 ID
                    crosswalk_label = f'횡단보도 {score:.2f}'
                    crosswalk_violation = "횡단보도 위반"
                else:
                    continue

                # 해당 레이블과 위반 사항 출력
                if cls == 0:
                    print(f'{braille_block_label}')
                    violation_type = braille_block_violation
                elif cls == 1:
                    print(f'{crosswalk_label}')
                    violation_type = crosswalk_violation

                # 위반 사항 기록
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                evidence_path = f"{evidence_dir}/frame_{frame_count}.jpg"
                cv2.imwrite(evidence_path, frame)
                latitude, longitude = 37.5665, 126.9780 # 임의의 값

                violation = {
                    "timestamp": timestamp,
                    "violation_type": violation_type,
                    "score": score,
                    "evidence_path": evidence_path,
                    "latitude": latitude,
                    "longitude": longitude
                }
                violations.append(violation)
                print(violation)

    # 결과 화면에 표시
    cv2.imshow('Crosswalk and Braille Block Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    if frame_count % capture_interval == 0:
        print(f'Processed {frame_count} frames')

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
