import time
import math
import cv2
import argparse
import numpy as np
import pkg_resources.py2_warn

# CALIBRATION FOR EACH VIDEO 

confid = 0.5
thresh = 0.5
angle_factor = 0.3 # default = 0.8 set closer to 1 for the more vertical the angle the camera is at
H_zoom_factor = 1.2

# Mask functions

def display_header():
    cv2.line(FR,(0,H+1),(FW,H+1),(0,0,0),2)
    cv2.putText(FR, "Social Distancing Analyser & Mask Detector ", (150, H+60),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(FR, (20, H+80), (510, H+180), (100, 100, 100), 2)
    cv2.putText(FR, "Connecting lines shows closeness among people. ", (30, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
    cv2.putText(FR, "-- YELLOW: CLOSE", (50, H+90+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 170), 2)
    cv2.putText(FR, "--    RED: VERY CLOSE", (50, H+40+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "--    PINK: Pathway for Calibration", (50, 150),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,105,255), 1)

    cv2.rectangle(FR, (535, H+80), (1060, H+140+40), (100, 100, 100), 2)
    cv2.putText(FR, "Bounding box shows the level of risk to the person.", (545, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
    cv2.putText(FR, "-- DARK RED: HIGH RISK", (565, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
    cv2.putText(FR, "--   ORANGE: LOW RISK", (565, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)
    cv2.putText(FR, "--    GREEN: SAFE", (565, H+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

def display_footer():
    cv2.putText(FR, tot_str, (10, H +25),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(FR, safe_str, (200, H +25),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 170, 0), 2)
    cv2.putText(FR, low_str, (380, H +25),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 120, 255), 2)
    cv2.putText(FR, high_str, (630, H +25),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 150), 2)

def get_predection_mask(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs[0])
    end = time.time()

    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    num_class_0 = 0
    num_class_1 = 0

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            #print(detection)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                if(classID==0):
                  num_class_0 +=1
                elif(classID==1):
                  num_class_1 +=1  

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
    if(num_class_0>0 or num_class_1>0):
        index= num_class_0/(num_class_0+num_class_1)
    else:
      index=-1  
    #print(index,)    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the 
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]])
            #print(boxes)
            #print(classIDs)      
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color=(69, 60, 90), thickness=2)
            cv2.rectangle(image, (x, y-5), (x+62, y-15), color, cv2.FILLED)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1)      
    return image

def get_labels(label_dir):
    return open(label_dir).read().split('\n')
      
def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

# DISTANCE FUNCTIONS

def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0

ap = argparse.ArgumentParser()
ap.add_argument("-l","--label_dir",type=str,
  default="obj.names")
ap.add_argument("-C","--CFG",type=str,
  default="yolov3-tiny_obj_train.cfg")
ap.add_argument("-w","--weights",type=str,
  default="yolov3-trained.weights")
ap.add_argument("-c", "--confthres", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-n", "--nmsthres", type=float, default=0.1,
    help="minimum probability to filter non-max suppresion")
ap.add_argument("-i","--input",required=False,
    help="path to video file")
ap.add_argument("-o","--output",required=False,
    help="path to video file")
args = vars(ap.parse_args())
print(args["CFG"])
Lables=get_labels(args["label_dir"])
nets = load_model(args["CFG"],args["weights"])
input_path =  args["input"]
output_path = args["output"]
confthres = args["confthres"]
nmsthres =  args["nmsthres"]

labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

#weightsPath = "./yolov3.weights"
#configPath = "./yolov3.cfg"

###### use this for faster processing (caution: slighly lower accuracy) ###########

weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
configPath = "./yolov3-tiny_obj_train.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0
#vs = cv2.VideoCapture(vid_path)
vs = cv2.VideoCapture(0)  ## USe this if you want to use webcam feed
#writer = None
(W, H) = (None, None)

fl = 0
q = 0
while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:] = col
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
    Colors=[(42,236,42),(0,0,255)]
    frame=get_predection_mask(frame,nets,Lables,Colors)
    if(len(idxs)==0):
    	cv2.imshow("frame",frame)
    	if cv2.waitKey(1) & 0xFF==ord('q'):
    		break
    if len(idxs) > 0:

        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []

        for i in idf:
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])

            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0

        for i in idf:
            
            display_header()
            
            tot_str = "TOTAL COUNT: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "LOW RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)

            display_footer()

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
        FR[0:H, 0:W] = frame
        frame = FR
        cv2.imshow('Social distancing analyser', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
        	break
        	
vs.release()
