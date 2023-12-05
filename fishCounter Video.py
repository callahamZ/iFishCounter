import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *

# Video
cap = cv2.VideoCapture("video.mp4")
#
model = YOLO("last.pt")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.4)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    sukses, img = cap.read()

    hasil = model.predict(img, stream=True, iou=0.4, conf=0.5)
    detections = np.empty((0, 5))

    for h in hasil:
        boxes = h.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0]) * 100)
            #if conf > 50:
            cvzone.cornerRect(img, (x1, y1, w, h))

            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))
            #indexclass = int(box.cls[0])
            cvzone.putTextRect(img, f'Ikan {conf}%', (max(0, x1), max(20, y1)), scale=1, thickness=1)
            #Max() untuk membatasi agar tidak melewati boundary

    resultsTracker = tracker.update(detections)

    #cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        #cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        #cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
        if totalCount.count(id) == 0:
            totalCount.append(id)
            #cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    #out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
