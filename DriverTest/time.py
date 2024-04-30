import datetime
import time

# set_time = datetime.datetime(2021, 11, 4, 17, 53, 45)
# now = datetime.datetime.now()
# dif = set_time - now
# print("Date: {:%Y, %m, %d}".format(now))
# print("Time: {:%H/%M/%S}".format(now))
# print("차이:", dif)

def sleep_ckecking(start):
    now = time.time()
    time.sleep(1)
    print(now - start)
    if now - start > 5:
        print("wraning")
    else:
        sleep_ckecking(start)
    if now - start > 10:
        print("finish")
        return

cnt = 0
while(True):
    now = time.time()
    cnt += 1
    if cnt == 5:
        start = time.time()  # 시작 시간 저장
        sleep_ckecking(start)

