# 실시간 운전자 동작 감지
* 2021년 숭실대학교 경진대회 최우수상 수상

## 개요
교통사고는 최근 5년간 1,116,035건이 발생했습니다.   
한국도로공사에 따르면 전방주시태만이 가장 높은 사고 원인이라고 밝혔습니다.  
현재 자율주행의 보편화에 비해 운전자 상태 감지는 아직 많이 부족한 상황입니다.  
따라서 운전자의 졸음 감지와 전방 주시 태만의 원인이 되는 물건을 감지합니다.  

## 코드 구성
  - eye_CNN.py : CNN 모델 및 학습 결과
  - Dection_Main.py : 실시간 운전자의 졸음 감지 및 핸드폰 감지하는 코드

## 사용 기술
* Dlib
* Hog Algorithm
* YoloV3
* CNN
* DeepLearning
* Tensorflow
* Keras

## 결과
![phone_result](https://user-images.githubusercontent.com/62223905/152672174-a140b7b4-3137-4e23-8da2-7cbda34259b0.jpg)

## References
[1] Z. Cao, G. Hidalgo, T. Simon, S-E. Wei, and Y. Sheikh. "OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields." In CVPR, 2017.   
[2] Hung, D. Huu, and H. Saito. "Fall detection with two cameras based on occupied area." Proc. of 18th Japan-Korea Joint Workshop on Frontier in Computer Vision. 2012.   
[3] B. Kwolek, and M. Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer", Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607.   
[4] F.Song, X.Tan, X.Liu and S.Chen, Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients, Pattern Recognition, 2014.   
[5] Iparaskev, Simple Blink Detector, https://github.com/iparaskev/simple-blink-detector.   
[6] Kairess, Eye Blink Detector, https://github.com/kairess/eye_blink_detector.    
[7] P. Chandran, D. Bradley, M. Gross, T. Beeler, "Attention-Driven Cropping for Very High Resolution Facial Landmark Detection", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.   
[8] K. -N. Park, “A Facial Morphing Method Using Delaunay Triangle of Facial Landmarks”, 디지털콘텐츠학회논문지, Journal of Digital Contents Society Vol. 19, No. 1, pp. 213-220, Jan. 2018.   
[9] J. Redmon, and A. Farhadi, “Yolov3: An incremental improvement”, arXiv preprint arXiv:1804.02767, 2018.   
[10] J.-Y. Kim, and S.-H. Kim, “Deep Learning based Object Detection Method and its Application for Intelligent Transport Systems,” Institute of Control, Robotics and Systems (ICROS), 27(12), pp. 1016-1022, Dec. 2021    

## APPENDIX
#### [추가 필요 파일]
  - yolov3.weights
  - yolov3.cfg
  - coco.names
  - shape_predictor_68_face_landmarks.dat
