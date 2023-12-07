# The prediction here checks for the face yawn, eyes colsed or not and this is for the model build in CNN_Model_Building.py
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import dlib
from imutils import face_utils
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
IMG_SIZE = 200

# Load the pre-trained cascade classifiers for eyes
face_cascade = cv2.CascadeClassifier('C:/Users/admin/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/admin/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_eye.xml')

model = tf.keras.models.load_model("D:/CNN-Drowsiness_logic/edge_model_eye/recognition_model_2.h5",compile=False)

# Load the facial landmark indices for the eyes
(l_start, l_end) = (42, 47)  # Left eye landmarks indices
(r_start, r_end) = (36, 41)  # Right eye landmarks indices

#cap = cv2.VideoCapture(0)  # 0  to access the camera "rtsp://admin:Unidad123@192.168.1.44/1"
#cap = cv2.VideoCapture('rtsp://admin:Unidad123@192.168.2.116/1', cv2.CAP_FFMPEG)  # 0 to access the camera
#cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
#cap = cv2.VideoCapture('rtsp://admin:Unidad123@192.168.2.123/1')
label = ['Closed','Open']
closeCounter = 0
while (True):
    cap = cv2.VideoCapture('rtsp://admin:Unidad123@192.168.2.44/1', cv2.CAP_FFMPEG)  # 0 to access the camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    ret, frame = cap.read()
    #print(type(frame))
    #img_array = cv2.imread(frame, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (640, 480))  # width,height
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    edge_img = cv2.Canny(gray, threshold1=30, threshold2=100)
    #resized_array = cv2.resize(edge_img, (IMG_SIZE, IMG_SIZE))
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # loop over the detected faces

    print('No. of faces detected: ', len(faces))
    for (x, y, w, h) in faces:
        #.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = frame[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        roi_edge_face = edge_img[y:y + h, x:x + w]
        #Do the nparray conversion and size change here
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        pred = []
        for (ex,ey,ew,eh) in eyes:
            roi_eye = roi_edge_face[ey:ey + eh, ex:ex + ew]
            
            resized_image = cv2.resize(roi_eye, (IMG_SIZE, IMG_SIZE))
            temp = np.argmax(model.predict([resized_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)]))
            pred.append(temp)
            print(temp)
            cv2.imshow("eyes", resized_image)        
        if sum(pred) <=1:
            closeCounter += 1
            if closeCounter > 20:
                cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            closeCounter = 0
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame with the prediction
    cv2.imshow("Drowsiness Prediction", frame)
        
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()



