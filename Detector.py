
import cvlib    # high level module, uses YOLO model with the find_common_objects method
import cv2      # image/video manipulation, allows us to pass frames to cvlibFrame=0
import pafy
import os
if os.name == 'posix':
    f = "https://www.youtube.com/watch?v=zu6yUYEERwA"
if os.name == 'nt':
    f = open("live.txt", "r")
    f= f.read()

    
print(f)
video = pafy.new(f)
best = video.getbest(preftype="mp4")

Frame=0
perFrame=11
yolo='yolov5'
confidence=.65
gpu=False
cap = cv2.VideoCapture(best.url)
while cap.isOpened():
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    if Frame%perFrame== 0:
        bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence, enable_gpu=gpu)
            
        if 'person' in labels:
            marked_frame = cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)
            cv2.imshow('Marked',marked_frame)
    else:
        unmarked = frame
        if bbox:
            unmarked=cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)
        cv2.imshow('Marked',unmarked)
    Frame+=1
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        
        break
    

cap.release()
cv2.destroyAllWindows()