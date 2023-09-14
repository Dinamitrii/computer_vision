import cv2
import dlib
from imutils import face_utils
from pygame import mixer

threshold = 6

mixer.init()
sound = mixer.Sound("/usr/share/sounds/sound-icons/xylofon.wav")

detection_list = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

capture = cv2.VideoCapture(0)


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


while True:
    # Capturing image from webcam
    _, image = capture.read()
    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # Getting face into the image of webcam
    rects = detector(gray, 0)

    # For each detected face finding the landmarks
    for (i, rect) in enumerate(rects):
        # Here making the prediction and converting into numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Drawing our image with all found coordinate points
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        leye_38 = shape[37]
        leye_39 = shape[38]
        leye_41 = shape[40]
        leye_42 = shape[41]

        reye_44 = shape[43]
        reye_45 = shape[44]
        reye_47 = shape[46]
        reye_48 = shape[47]

        detection_list.append(
            (dist(leye_38, leye_42) + dist(leye_39, leye_41) + dist(reye_44, reye_48) + dist(reye_45,
                                                                                             reye_47)) / 4 < threshold)

        if len(detection_list) > 10:
            detection_list.pop(0)

        # If drowsiness detected
        if sum(detection_list) > 4:
            try:
                sound.play()
            except:
                pass

        else:
            try:
                sound.stop()
            except:
                pass

    # Showing the image
    cv2.imshow("Output", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
capture.release()
