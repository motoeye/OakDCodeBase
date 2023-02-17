from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import blobconverter
import RPi.GPIO as GPIO
import time
import heapq

def initializeLeds():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)


def LedOnFunc():        
    GPIO.output(11, GPIO.HIGH)

def LedOffFunc():        
    GPIO.output(11, GPIO.LOW)

def getLedStatus():
    # returns 0 if off or 1 if on
    GPIO.input(11)

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

FRAME_SIZE = (640,360)
DET_INPUT_SIZE = (300,300)
# model_name = "face-detection-retail-0004"
# model_name = "yolov4_tiny_coco_416x416"
# model_name = "yolo-v3-tf"
# model_name = "yolov5n_coco_416x416"
model_name = "yolov6n_coco_416x416"
zoo_type = "depthai"
blob_path = None

if model_name != None:
    nnBlobPath = blobconverter.from_zoo(
        name=model_name,
        shaves=6,
        zoo_type=zoo_type
    )


# Tiny yolo v3/4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase"
]

admissibleList = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","chair","sofa", "refrigerator"
]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)

spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

initializeLeds()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    flag =0
    flag2=0
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        if printOutputLayersOnce:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False;

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        if len(detections)==0:
            LedOffFunc()
        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # print(f"{label} : {detection.spatialCoordinates.z}mm")
            # if (label in admissibleList):
            #     if (int(detection.spatialCoordinates.z) < 10000) and (int(detection.spatialCoordinates.z)>60) :
            #         LedOnFunc()
            #         print(f"{label} : {detection.spatialCoordinates.z}mm")                         
            #     else:
            #         LedOffFunc()
            
            # elif label not in admissibleList:
            #     LedOffFunc()
            # print(int(detection.spatialCoordinates.z))

            #getLedStatus()
            #last try, this will work
            # if (int(detection.spatialCoordinates.z) < 10000) and (int(detection.spatialCoordinates.z)>60) :
            #     if (label in admissibleList): #human detected
            #         flag = 1
            #         LedOnFunc()
            #         print(f"{label} : {detection.spatialCoordinates.z}mm")
            #     else:
            #         if (label not in admissibleList) and getLedStatus()==1 and flag2==1 and flag==0: #nothing
            #             LedOffFunc()
            #         elif (label not in admissibleList) and getLedStatus()==1 and flag2==0: #fridge enters after human
            #             flag2=1
            #         elif (label not in admissibleList) and getLedStatus()==1 and flag2==1 and flag==1: #human leaves, fridge remains
            #             print("AAAAA")
            #             flag=0
            #             LedOffFunc()
            #         elif (label not in admissibleList) and getLedStatus()==0 and flag2==0: #fridge enters first
            #             flag2=1                     
            # else:
            #     LedOffFunc()
            #     flag2=0
            
            # works well except 1 condition nc-c-nc
            # if (int(detection.spatialCoordinates.z) < 10000) and (int(detection.spatialCoordinates.z)>60) :
            #     if (label in admissibleList): #human detected
            #         flag = 1
            #         LedOnFunc()
            #         print(f"{label} : {detection.spatialCoordinates.z}mm")
            #     else:
            #         if (label not in admissibleList) and flag==1 and flag2==1: #turn off when human leaves
            #             LedOffFunc()
            #             flag1=0
            #         elif (label not in admissibleList) and flag==1 and flag2==0: #enters when fridge joins human
            #             flag2=1
            #             if (flag==1 and flag2==1): #turn on when human joins fridge
            #                 LedOnFunc()
            #             else:
            #                 LedOffFunc()
            #                 flag=0
            #         elif (label not in admissibleList) and flag==0 and flag2==1: #remains off while fridge alone
            #             LedOffFunc()
            #         else: #happens when fridge detected
            #             flag2=1

            # else:
            #     LedOffFunc()
            #     flag2=0


            #new logic
            # if (int(detection.spatialCoordinates.z) < 10000) and (int(detection.spatialCoordinates.z)>60) :
            #     if (label in admissibleList):
            #         flag = 1
            #     if (label not in admissibleList):
            #         flag2 = 1
            #     if flag==0 and flag2==0:
            #         LedOffFunc()
            #     if flag==0 and flag2==1:
            #         flag2==0
            #         LedOffFunc()
            #     if flag==1 and flag2==0:
            #         LedOnFunc()
            #     if flag==1 and flag2==1:
            #         LedOnFunc()
            #     flag2=0
            # else:
            #     flag=0
            #     flag2=0
            #     LedOffFunc()

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break