# -*- coding: utf-8 -*-
import cv2
import random

# Parameters
TH1 = 0.5  # Confidence
TH2 = 0.25 # Non-maximum Suppression

# CNN model
config = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/yolov3.cfg'
model = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/yolov3.weights'
classLabels = 'C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/coco.names'

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

# main(동영상 저장과 경고 알림 추가)
if __name__ == "__main__":
    # Load CNN model trained over MS COCO DB
    net = cv2.dnn.readNetFromDarknet(config, model)
    layerNames = net.getLayerNames()
    layerOutputs = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]