
from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Define object classes for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True :
    ret, frame = cap.read()

    results = model(frame, stream=True) #we get each result one by one using 'stream=True' as asmd when results are produced hence reducing memory usage
    for r in results:
        boxes = r.boxes
        # print(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0),3)
            confidence = float((box.conf[0]*100)/100)
            confidence = round(confidence,2)
            print("Confidence", confidence)
            cls = int(box.cls[0])
            print("Class",cls)
            #text in frame
            org = [x1, y1]
            conf_org = [x2, y2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            cv2.putText(frame, classNames[cls], org, font, 1, color, 2)
            cv2.putText(frame, str(confidence), conf_org, font, 1, color, 2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()