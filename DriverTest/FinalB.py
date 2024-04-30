# module
import cv2
import random
import numpy as np
import dlib
from tensorflow import keras
from imutils import face_utils

# Parameters
TH1 = 0.5  # Confidence
TH2 = 0.25 # Non-maximum Suppression
IMG_SIZE = (34, 26)

# yolo file
config = 'C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/yolov3/yolov3.cfg'
model = 'C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/yolov3/yolov3.weights'
classLabels = 'C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/yolov3/coco.names'

# eyes file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/modul/shape_predictor_68_face_landmarks.dat')
cnn_model = keras.models.load_model("C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/modul/cnn_model.h5")

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet(model,config)

# from darknet.py
def fun_get_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    random.seed(1)
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}

def eye_calc(eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_Ebox = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)  # 정수 array로 변환

    eye_img = gray[eye_Ebox[1]:eye_Ebox[3], eye_Ebox[0]:eye_Ebox[2]]

    return eye_img, eye_Ebox

def eye_detect(gray, face):
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_Ebox_l = eye_calc(eye_points=shapes[36:42])
    eye_img_r, eye_Ebox_r = eye_calc(eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = cnn_model.predict(eye_input_l)
    pred_r = cnn_model.predict(eye_input_r)

    return pred_l, pred_r, eye_Ebox_l, eye_Ebox_r

# main
# pre = 0
if __name__ == "__main__":
    cnt = 0
    # YOLO NETWORK 재구성
    class_names = []
    with open(classLabels, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    class_colors = fun_get_colors(class_names)
    layer_names = YOLO_net.getLayerNames()
    reshapeNet = YOLO_net.getUnconnectedOutLayers().reshape(3, 1)
    output_layers = [layer_names[i[0] - 1] for i in reshapeNet]

    while True:
        # 웹캠 프레임
        ret, frame = VideoSignal.read()
        if not ret:
            break

        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facial = detector(gray)

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        classIDs = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > TH1:
                    # Object detected
                    centerX, centerY = int(detection[0] * w), int(detection[1] * h)
                    boxW, boxH = int(detection[2] * w), int(detection[3] * h)
                    # Rectangle coordinate
                    topleftX, topleftY = int(centerX - boxW / 2), int(centerY - boxH / 2)
                    boxes.append([topleftX, topleftY, boxW, boxH])
                    confidences.append(float(confidence))
                    classIDs.append(class_id)
        resultDet = cv2.dnn.NMSBoxes(boxes, confidences, TH1, TH2)

        # Visualization
        for i in resultDet:
            x, y, w, h = boxes[i]
            label = str(class_names[classIDs[i]])
            score = confidences[i]
            if "cell phone" in label:
                cv2.putText(frame, "Warning!", (40, 280), cv2.FONT_HERSHEY_TRIPLEX, 4.0, (0, 0, 255), 5)  # 입력 이미지, 입력 문구, 문구 시작 위치, 폰트, 글자크기, 글자색상, 글자굵기
            # 경계상자와 클래스 정보 이미지에 입력
            color = class_colors[class_names[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, color, 2)

        for face in facial:
            pred_l, pred_r, eye_Ebox_l, eye_Ebox_r = eye_detect(gray, face)

            # visualize
            state_l = '%.1f' if pred_l > 0.1 else '%.1f'
            state_r = '%.1f' if pred_r > 0.1 else '%.1f'

            state_l = state_l % pred_l
            state_r = state_r % pred_r

            # 1 frame = 0.35 sec
            if state_l == '0.0' and state_r == '0.0':
                cnt += 1
            if cnt > 4:
                cv2.putText(frame, "Warning!", (35, 280), cv2.FONT_HERSHEY_TRIPLEX, 4.0, (0, 0, 255), 5)
            # if state_l > '0.3' or state_r > '0.3':
            #     cnt = 0

            if state_l < '0.3' and state_r < '0.3':
                cv2.rectangle(frame, pt1=tuple(eye_Ebox_l[0:2]), pt2=tuple(eye_Ebox_l[2:4]), color=(0, 0, 255), thickness=2)
                cv2.rectangle(frame, pt1=tuple(eye_Ebox_r[0:2]), pt2=tuple(eye_Ebox_r[2:4]), color=(0, 0, 255), thickness=2)
            else:
                cnt = 0
                cv2.rectangle(frame, pt1=tuple(eye_Ebox_l[0:2]), pt2=tuple(eye_Ebox_l[2:4]), color=(255, 255, 255), thickness=2)
                cv2.rectangle(frame, pt1=tuple(eye_Ebox_r[0:2]), pt2=tuple(eye_Ebox_r[2:4]), color=(255, 255, 255), thickness=2)

            # # 눈 상태값 출력
            # cv2.putText(frame, state_l, tuple(eye_Ebox_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(frame, state_r, tuple(eye_Ebox_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Driver Detection", frame)

        if cv2.waitKey(100) > 0:
            break