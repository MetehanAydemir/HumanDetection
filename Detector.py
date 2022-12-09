
import cvlib    # high level module, uses YOLO model with the find_common_objects method
import cv2      # image/video manipulation, allows us to pass frames to cvlibFrame=0
import os

#Frame Counter    
Frame=0
#Detect period with Frame
perFrame=11

#Yolo version if you want faster detection for mobile you can use 'yolov4-tiny'
yolo='yolov4'
#Trashhold level for detection 
confidence=.65
#If you use local computer and you have Graphic card choose
gpu=False

####Video Capture source. 0 is webcam 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    ##Cv2 framing pictures to picture BGR format. Some model uses RGB format. Then you have convert with your format 
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    ##If statement for detection period
    if Frame%perFrame== 0:
        ###BBox bounding box contains object locations
        ###Labels contains predict person, bed etc.
        ###Conf is detect rate
         bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence, enable_gpu=gpu)
        
        if 'person' in labels:
            ### Draws bounding box and label name around detects
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
