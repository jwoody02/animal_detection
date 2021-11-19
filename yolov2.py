# import the necessary packages
import numpy as np
import sys
import argparse
import time
import cv2
import os
from collections import Counter
import os, os.path
isfile = os.path.isfile
join = os.path.join

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=False,
	help="base path to YOLO directory", default="yolo-coco/")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
ap.add_argument("-mi", "--multimage", required=False, help="path to images directory -- images will automatically be loaded")
args = vars(ap.parse_args())
if args['image'] is None and args['multimage'] is None:
    print("No image/image path given. Please provide an input for -i or -mi")
    sys.exit(0)
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "yolov3.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

animals_id = []
amount_f = []
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
def detectFromImage(image_arg):

    # load our input image and grab its spatial dimensions
    image = cv2.imread(image_arg)
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
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    def getName(id):
        return LABELS[classIDs[id]]
    #change if you have a different camera
    focal_length = 4.25

    nothing_here = True
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]

    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > args["confidence"]:

    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])


    			(centerX, centerY, width, height) = box.astype("int")


    			# use the center (x, y)-coordinates to derive the top and

    			nothing_here = False
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates, confidences,


                # prints info for each detected object
    			#print("[DEBUG] ["+str(LABELS[classID])+"] (x,y,w,h): "+str(x)+", "+str(y)+", "+str(width)+", "+str(height))

    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    print("\nResults (\033[4m"+str(image_arg)+"\033[0m)::")
    if nothing_here:
        print("\nNothing here")

    else:
        array_counter = dict(Counter(classIDs))
        for i in range(len(array_counter)):
            tostr = (str(array_counter).replace("{","")).replace("}","")
            name = LABELS[int((tostr.split(",")[i]).split(":")[0])]
            number = int((tostr.split(",")[i]).split(":")[1])
            print(name+" ("+str(number)+")")
    print(".................................................................\n")
    #        print(tostr.split(",")[i])


    #    print(array_counter)
    # UNCOMMENT TO PRESENT IMAGE
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    #
    # # ensure at least one detection exists
    # if len(idxs) > 0:
    # 	# loop over the indexes we are keeping
    # 	for i in idxs.flatten():
    # 		# extract the bounding box coordinates
    # 		(x, y) = (boxes[i][0], boxes[i][1])
    # 		(w, h) = (boxes[i][2], boxes[i][3])
    #
    # 		# draw a bounding box rectangle and label on the image
    # 		color = [int(c) for c in COLORS[classIDs[i]]]
    # 		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    # 		text = "{}: {:.2f}%".format(LABELS[classIDs[i]], (confidences[i]*100))
    # 		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    # 			0.5, color, 2)
    #
    # # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

def detectAndPresent(image_arg):
    # shows the image w/ bounding boxes(instead of just printing)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(image_arg)
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
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    def getName(id):
        return LABELS[classIDs[id]]
    #change if you have a different camera
    focal_length = 4.25

    nothing_here = True
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]

    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > args["confidence"]:

    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])


    			(centerX, centerY, width, height) = box.astype("int")


    			# use the center (x, y)-coordinates to derive the top and

    			nothing_here = False
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates, confidences,


                # prints info for each detected object
    			#print("[DEBUG] ["+str(LABELS[classID])+"] (x,y,w,h): "+str(x)+", "+str(y)+", "+str(width)+", "+str(height))

    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    print("\nResults (\033[4m"+str(image_arg)+"\033[0m)::")
    if nothing_here:
        print("\nNothing here")

    else:
        array_counter = dict(Counter(classIDs))
        for i in range(len(array_counter)):
            tostr = (str(array_counter).replace("{","")).replace("}","")
            name = LABELS[int((tostr.split(",")[i]).split(":")[0])]
            number = int((tostr.split(",")[i]).split(":")[1])
            print(name+" ("+str(number)+")")
    #        print(tostr.split(",")[i])


    #    print(array_counter)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
    	for i in idxs.flatten():
    		# extract the bounding box coordinates
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])

    		# draw a bounding box rectangle and label on the image
    		color = [int(c) for c in COLORS[classIDs[i]]]
    		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    		text = "{}: {:.2f}%".format(LABELS[classIDs[i]], (confidences[i]*100))
    		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)




if args['image'] is None:
    # multimage con
    print("[INFO] multi image toggled")
    # print(len([name for name in os.listdir('.') if os.path.isfile(name)]))
    directory = args['multimage']
    number_of_files = sum(1 for item in os.listdir(directory) if isfile(join(directory, item)))
    print("[!] Loaded \033[4m"+str(number_of_files)+"\033[0m images")
    print("[!] Expected to take \033[4m"+str(0.5*number_of_files)+"\033[0m seconds")
    for filename in os.listdir(directory):
        # just add compatible image ending here if you have any
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            # print("Loading File: "+directory+"/"+filename)
            detectFromImage(directory+"/"+filename)
elif args['multimage'] is None:
    print("[INFO] single image toggled")
    # detectFromImage(path) provides visual and console output
    detectFromImage(args['image'])










#
# # load our input image and grab its spatial dimensions
# image = cv2.imread(args["image"])
# (H, W) = image.shape[:2]
#
# # determine only the *output* layer names that we need from YOLO
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# # construct a blob from the input image and then perform a forward
# # pass of the YOLO object detector, giving us our bounding boxes and
# # associated probabilities
# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
# 	swapRB=True, crop=False)
# net.setInput(blob)
# start = time.time()
# layerOutputs = net.forward(ln)
# end = time.time()
#
# # show timing information on YOLO
# print("[INFO] YOLO took {:.6f} seconds".format(end - start))
#
# # initialize our lists of detected bounding boxes, confidences, and
# # class IDs, respectively
# boxes = []
# confidences = []
# classIDs = []
# print("[INFO] Assuming Focal Length as 4.25")
# def getName(id):
#     return LABELS[classIDs[id]]
# #change if you have a different camera
# focal_length = 4.25
#
# nothing_here = True
# # loop over each of the layer outputs
# for output in layerOutputs:
# 	# loop over each of the detections
# 	for detection in output:
# 		# extract the class ID and confidence (i.e., probability) of
# 		# the current object detection
# 		scores = detection[5:]
# 		classID = np.argmax(scores)
# 		confidence = scores[classID]
#
# 		# filter out weak predictions by ensuring the detected
# 		# probability is greater than the minimum probability
# 		if confidence > args["confidence"]:
#
# 			# scale the bounding box coordinates back relative to the
# 			# size of the image, keeping in mind that YOLO actually
# 			# returns the center (x, y)-coordinates of the bounding
# 			# box followed by the boxes' width and height
# 			box = detection[0:4] * np.array([W, H, W, H])
#
#
# 			(centerX, centerY, width, height) = box.astype("int")
#
#
# 			# use the center (x, y)-coordinates to derive the top and
#
# 			nothing_here = False
# 			x = int(centerX - (width / 2))
# 			y = int(centerY - (height / 2))
#
# 			# update our list of bounding box coordinates, confidences,
#
# 			print("[DEBUG] ["+str(LABELS[classID])+"] (x,y,w,h): "+str(x)+", "+str(y)+", "+str(width)+", "+str(height))
#
# 			boxes.append([x, y, int(width), int(height)])
# 			confidences.append(float(confidence))
# 			classIDs.append(classID)
#
#
#
# # apply non-maxima suppression to suppress weak, overlapping bounding
# # boxes
# print("============RESULTS============\n\n")
# if nothing_here:
#     print("\nNothing here")
#
# else:
#     array_counter = dict(Counter(classIDs))
#     for i in range(len(array_counter)):
#         tostr = (str(array_counter).replace("{","")).replace("}","")
#         name = LABELS[int((tostr.split(",")[i]).split(":")[0])]
#         number = int((tostr.split(",")[i]).split(":")[1])
#         print(name+" ("+str(number)+")")
# #        print(tostr.split(",")[i])
#
#
# #    print(array_counter)
# idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
#
# # ensure at least one detection exists
# if len(idxs) > 0:
# 	# loop over the indexes we are keeping
# 	for i in idxs.flatten():
# 		# extract the bounding box coordinates
# 		(x, y) = (boxes[i][0], boxes[i][1])
# 		(w, h) = (boxes[i][2], boxes[i][3])
#
# 		# draw a bounding box rectangle and label on the image
# 		color = [int(c) for c in COLORS[classIDs[i]]]
# 		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
# 		text = "{}: {:.2f}%".format(LABELS[classIDs[i]], (confidences[i]*100))
# 		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
# 			0.5, color, 2)
#
# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
