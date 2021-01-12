# Import the necessary packages
import numpy as np
import cv2
import os
import tampering_detection as tmp
import time


def load_objectDetector():

    # Load the COCO class labels the YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # Derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

    # Load the  YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return LABELS, net, ln


def initialise():

    while True:
        video_path = input("Enter the video path :")
        vs = cv2.VideoCapture(video_path)
        ret, frame = vs.read()
        if ret == False:
            print("invalid video path")
        else:
            break
    Output_path = video_path[:-4] + "output.avi"
    writer = None
    (W, H) = (None, None)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame)
    return vs, writer, (W, H), fgbg, fgmask, frame, Output_path


def main():
    LABELS, net, ln = load_objectDetector()
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    vs, writer, (W, H), fgbg, fgmask, frame, Output_path = initialise()
    tracker = cv2.TrackerCSRT_create()

    tampcount = 0
    discount = 0
    colcount = 0
    bbox = cv2.selectROI(frame, False)
    (H, W) = frame.shape[:2]

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    framecount = 0

    # Loop over the frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        framecount = framecount +1
        isTampered = False
        isTampered = tmp.detect_tampering(fgbg, frame)
        if isTampered == True:
            if framecount < 60:
                tampcount = tampcount + 1
            if tampcount == 2:
                print("Tampering detected")
                cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # mailServer = smtplib.SMTP('smtp.gmail.com', 587)
                # print('>>>>>>')
                # mailServer.starttls()
                # mailServer.login('shravanthi.musti@gmail.com','bhasker98')
                # mailServer.sendmail('shravanthi.musti@gmail.com', 'sravani055@gmail.com','Alert')
                # print(" \n Sent!")
                # mailServer.quit()
                tampcount = 0
        # If the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # If the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # Initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # Loop over each of the layer outputs
        for output in layerOutputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # Scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            if (c1 == 0) and (c2 == 0):
                c1 = p1[0]
                c2 = p1[1]
            if (c3 == 0) and (c4 == 0):
                c3 = p2[0]
                c4 = p2[1]

        else:
            # Tracking failure
            cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 170, 50), 2);

        s1 = p1[0] - c1
        s2 = p1[1] - c2
        s3 = p2[0] - c3
        s4 = p2[1] - c4
        if (s1 > 45) or (s2 > 45):
            if framecount < 60:
                discount = discount + 1
            if discount == 4:
                print("Displacement detected")
                cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # mailServer = smtplib.SMTP('smtp.gmail.com', 587)
                # print('>>>>>>')
                # mailServer.starttls()
                # mailServer.login('shravanthi.musti@gmail.com', 'bhasker98')
                # mailServer.sendmail('shravanthi.musti@gmail.com', 'sravani055@gmail.com', 'Alert')
                # print(" \n Sent!")
                # mailServer.quit()
                discount = 0
        elif (s3 > 45) or (s4 > 45):
            if framecount < 60:
                discount = discount + 1
            if discount == 4:
                print("Displacement detected")
                cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # mailServer = smtplib.SMTP('smtp.gmail.com', 587)
                # print('>>>>>>')
                # mailServer.starttls()
                # mailServer.login('shravanthi.musti@gmail.com', 'bhasker98')
                # mailServer.sendmail('shravanthi.musti@gmail.com', 'sravani055@gmail.com', 'Alert')
                # print(" \n Sent!")
                # mailServer.quit()
                discount = 0
        # ensure at least one detection exists
        if len(idxs) > 0:
            # Loop over the indexes we are keeping
            for i in idxs.flatten():
                cv2.imshow("frame", frame)
                if LABELS[classIDs[i]] == "person" or LABELS[classIDs[i]] == "knife":
                    # Extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # Draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if LABELS[classIDs[i]] == 'knife':
                        cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        print('knife Detected')
                    if ((abs(w - h) < 11) or (w > 360) or (h > 360)) and (LABELS[classIDs[i]] == "person"):
                        if framecount < 60:
                            colcount = colcount + 1
                        if colcount == 4:
                            cv2.putText(frame, "Unusual activity detected", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                            print("Collision detected")
                            cv2.putText(frame, "Collision detected", (10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                            # mailServer = smtplib.SMTP('smtp.gmail.com', 587)
                            # print('>>>>>>')
                            # mailServer.starttls()
                            # mailServer.login('shravanthi.musti@gmail.com', 'bhasker98')
                            # mailServer.sendmail('shravanthi.musti@gmail.com', 'sravani055@gmail.com', 'Alert')
                            # print(" \n Sent!")
                            # mailServer.quit()
                            colcount = 0
        # Check if the video writer is None
        if writer is None:
            # Initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(Output_path, fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        # Write the output frame to disk
        kframe = frame
        cv2.imshow('frame', kframe)
        cv2.waitKey(1)
        writer.write(frame)
        if framecount == 60:
            framecount = 0

    writer.release()
    vs.release()


main()
