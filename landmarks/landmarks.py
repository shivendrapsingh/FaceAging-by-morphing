from flask import Flask, Response, json, request
from imutils import face_utils
import imutils
import dlib
import cv2
import landmarks.lmconfig
import os


def getcoordinates(photo):
    coordinates = []

    # image = request.files.get('image')
    image = photo
    og = cv2.imread(photo)
    print(str(image))
    shape_predictor = os.path.join('landmarks', 'shape_predictor_81_face_landmarks.dat')
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image)
    # image = imutils.resize(image, height=1920, width=1080)
    image = cv2.resize(image, (landmarks.lmconfig.rwidth, landmarks.lmconfig.rheight))
    reference = cv2.resize(og, (landmarks.lmconfig.rwidth, landmarks.lmconfig.rheight))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            coordinates.append((float(x), float(y)))
            # print((x, y))

    # show additional 9 landmarks
    cv2.circle(image, (0, 0), 3, (0, 0, 255), -1)
    cv2.circle(image, (0, landmarks.lmconfig.height), 3, (0, 0, 255), -1)
    cv2.circle(image, (landmarks.lmconfig.width, landmarks.lmconfig.height), 3, (0, 0, 255), -1)
    cv2.circle(image, (landmarks.lmconfig.width, 0), 3, (0, 0, 255), -1)
    cv2.circle(image, (landmarks.lmconfig.halfwidth, 0), 3, (0, 0, 255), -1)
    cv2.circle(image, (0, landmarks.lmconfig.halfheight), 3, (0, 0, 255), -1)
    cv2.circle(image, (landmarks.lmconfig.halfwidth, landmarks.lmconfig.height), 3, (0, 0, 255), -1)
    cv2.circle(image, (landmarks.lmconfig.width, landmarks.lmconfig.halfheight), 3, (0, 0, 255), -1)

    # add additional 9 landmarks coordinates
    coordinates.append((0, 0))
    coordinates.append((0, landmarks.lmconfig.height))
    coordinates.append((landmarks.lmconfig.width, landmarks.lmconfig.height))
    coordinates.append((landmarks.lmconfig.width, 0))
    coordinates.append((landmarks.lmconfig.halfwidth, 0))
    coordinates.append((0, landmarks.lmconfig.halfheight))
    coordinates.append((landmarks.lmconfig.halfwidth, landmarks.lmconfig.height))
    coordinates.append((landmarks.lmconfig.width, landmarks.lmconfig.halfheight))

    # save the output image with the face detections + facial landmarks and return reference image.
    # cv2.imwrite(os.path.join('static', 'landmarks.jpg'), image)
    if photo != "sample.jpg":
        cv2.imwrite(os.path.join('static', 'og.jpg'), reference)
    # cv2.waitKey(0)

    return json.dumps({'coordinates': coordinates}, sort_keys=False, indent=4, separators=(',', ': '))
