import cv2
import random
import numpy as np

# Parameters
TH1 = 0.5  # Confidence
TH2 = 0.25 # Non-maximum Suppression

# config = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/yolov3.cfg'
# model = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/yolov3.weights'
# classLabels = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/coco.names'

config = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/fifth/yolo-obj.cfg'
model = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/fifth/yolo-obj_final.weights'
classLabels = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/fifth/obj.names'

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet(model, config)

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


if __name__ == "__main__":
    # YOLO NETWORK 재구성
    class_names = []
    # output_layers = []
    with open(classLabels, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    class_colors = fun_get_colors(class_names)
    layer_names = YOLO_net.getLayerNames()
    reshapeNet = YOLO_net.getUnconnectedOutLayers().reshape(1, 1)
    output_layers = [layer_names[i[0] - 1] for i in reshapeNet]

    while True:
        # 웹캠 프레임
        ret, frame = VideoSignal.read()
        h, w, c = frame.shape

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
        #
        cv2.imshow("YOLOv3", frame)

        if cv2.waitKey(100) > 0:
            break