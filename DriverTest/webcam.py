import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, img_ori = cap.read()
    if ret == False:
        continue
    cv2.imshow("cam", img_ori)
    #종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()