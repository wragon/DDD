# 실시간 운전자 동작 감지
* 2021년 숭실대학교 경진대회 최우수상 수상

## 개요
교통사고는 최근 5년간 1,116,035건이 발생했습니다.   
한국도로공사에 따르면 전방주시태만이 가장 높은 사고 원인이라고 밝혔습니다.  
현재 자율주행의 보편화에 비해 운전자 상태 감지는 아직 많이 부족한 상황입니다.  
따라서 운전자의 졸음 감지와 전방 주시 태만의 원인이 되는 물건을 감지합니다.  

## 코드 구성
  - eye_video.ipynb : 운전자의 졸음 감지 코드
  - yolo_video.ipynb : 핸드폰을 감지하는 코드
  - final.ipynb : 위의 두가지 코드를 합친 완성본 코드

## 사용 기술
* Dlib
* Hog Algorithm
* YoloV3
* CNN
* DeepLearning
* Tensorflow
* Keras

## 결과
![image](https://user-images.githubusercontent.com/62223905/136493433-2f3265ea-dd02-430a-95ad-c1de9b636c97.png)

## References
[1]	공공데이터 사이트, “도로교통공단_최근 5년 교통사고 통계”.  
[2]	공공데이터 사이트, “경찰청_졸음운전 교통사고 현황”.  
[3]	HMG TV, “현대자동차 그룹 TV방송 채널”, https://tv.hmgjournal.com/MediaCenter/News/Group-Broadcast/news-02-190722.blg.  
[4]	동아 닷컴 뉴스, ”현대모비스, 2021년 운전자 얼굴·동공 인식하는 ‘부주의 경보장치’ 양산”, https://www.donga.com/news/article/all/20190714/96473658/3.  	  
[5]	BIZION,“Brain4cars”, http://www.bizion.com/bbs/board.php?bo_table=tech&wr_id=257&page=7.  
[6]	안병태, “Study for Drowsy Driving Detection & Prevention System”, KoreaScience, Vol. 8. No. 3, pp. 193-198, 2018.  
[7]	F.Song, X.Tan*,S.chen, Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients, Pattern Recognition, (2014) 2825-2838.  
[8]	박경남, “A Facial Morphing Method Using Delaunay Triangle of Facial Landmarks”, 디지털콘텐츠학회논문지, Journal of Digital Contents Society Vol. 19, No. 1, pp. 213-220, Jan. 2018.  
[9]	eye_blink_detector, https://github.com/kairess/eye_blink_detector.  
[10]	J Redmon, A Farhadi, “Yolov3: An incremental improvement”, arXiv preprint arXiv:1804.02767, 2018.  
[11]	YOLO_detector, https://github.com/seongheum-ssu/2021-1-2150054901/blob/main/exp/2021-05-13.  
[12]	Cao, Zhe, et al. "OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields." arXiv preprint arXiv:1812.08008 (2018).  

## APPENDIX
#### [추가 필요 파일]
  - yolov3.weights
  - yolov3.cfg
  - coco.names
  - shape_predictor_68_face_landmarks.dat
