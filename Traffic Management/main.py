#let's start
# importing OpenCV lib.
import cv2
# importing tracker
from tracker import *

# creating tracker object
tracker = EuclideanDistTracker()

# importing video
cap = cv2.VideoCapture("Road.mp4")

# method for object detection from Stable camera, (isolates moving objects)
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)
# for each frame
while True:
    # ret holds binary value, True if frame is found
    ret, frame = cap.read()
    # if there are frames, else end process
    if ret==True:
        height, width, _ = frame.shape
    else:
        cap.release()
        cv2.destroyAllWindows()
        print("Co-ordinates printed.\n \n End of the video...")



    # best Region of interest for tracking (for fast detection)
    roi = frame[300: 510,550: 950]

    # here, object detection takes place
    # mask screen gives us idea of how mmoving object is represented in R.O.I.
    mask = object_detector.apply(roi)
    # binary parameter (shows black and white screen) i.e. two possibilities moving/not moving
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # contours gives us boundry around white spaces i.e moving object.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # array for detection co-ordinates
    detections = []
    for cnt in contours:
        # calculating area and removing small elements
        area = cv2.contourArea(cnt)
        if area > 550:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            # adding co-ordinates
            detections.append([x, y, w, h])

    # object Tracking starts here
    # unique ID for each box
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # outputs
    cv2.imshow("roi", roi)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)


    # for exiting the program , 27 refers to escape key
    key = cv2.waitKey(30)
    if key == 27:
        break


# releasing the video capture object
cap.release()
#doing a bit of cleanup
cv2.destroyAllWindows()
