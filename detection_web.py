import cv2
from ultralytics import YOLO

# YOLO 학습된 모델 로드
model = YOLO("C:/Parking Detection/best.pt")

# 웹캠 열기
cap = cv2.VideoCapture(0)

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
            x1, y1, x2, y2 = box  # 바운딩 박스 좌표

            if score >= 0.5:  # 신뢰도가 0.5 이상일 때
                if cls == 0:  # 점자 블록 클래스
                    label = f'Block {score:.2f}'
                    color = (0, 255, 0)
                elif cls == 1:  # 횡단보도 클래스
                    label = f'Crosswalk {score:.2f}'
                    color = (0, 0, 255)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 결과 화면에 표시
    cv2.imshow('Braille Block and Crosswalk Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
