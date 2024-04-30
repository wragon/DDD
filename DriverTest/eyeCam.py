import numpy as np
import cv2
import dlib
from tensorflow import keras
from imutils import face_utils
import time

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/user/Desktop/Junyong/PycharmProjects/DriverDetection/modul/shape_predictor_68_face_landmarks.dat')

cnn_model = keras.models.load_model("cnn_model.h5")

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



VideoSignal = cv2.VideoCapture(0)
# pre = 0
# main(동영상 저장과 경고 알림 추가)
if __name__ == "__main__":
    cnt = 0
    while True:
        ret, frame = VideoSignal.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facial = detector(gray)

        # # time check
        # now = time.time()
        # print(now - pre)
        # pre = now

        for face in facial:
            pred_l, pred_r, eye_Ebox_l, eye_Ebox_r = eye_detect(gray, face)

            # visualize
            state_l = '%.1f' if pred_l > 0.1 else '%.1f'
            state_r = '%.1f' if pred_r > 0.1 else '%.1f'

            state_l = state_l % pred_l
            state_r = state_r % pred_r

            # 1 frame = 0.2 sec
            if state_l == '0.0' and state_r == '0.0':
                cnt += 1
            if cnt > 7:
                cv2.putText(frame, "Warning!", (35, 280), cv2.FONT_HERSHEY_TRIPLEX, 4.0, (0, 0, 255), 5)
            if state_l > '0.3' or state_r > '0.3':
                cnt = 0

            cv2.rectangle(frame, pt1=tuple(eye_Ebox_l[0:2]), pt2=tuple(eye_Ebox_l[2:4]), color=(255, 255, 255),
                          thickness=2)
            cv2.rectangle(frame, pt1=tuple(eye_Ebox_r[0:2]), pt2=tuple(eye_Ebox_r[2:4]), color=(255, 255, 255),
                          thickness=2)

            cv2.putText(frame, state_l, tuple(eye_Ebox_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, state_r, tuple(eye_Ebox_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Eyes", frame)

        if cv2.waitKey(100) > 0:
            break