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


IMG_SIZE = 200

# Load the pre-trained cascade classifiers for eyes
model = tf.keras.models.load_model("D:/CNN-Drowsiness_logic/edge_model/recognition_model_2.h5",compile=False)


cap = cv2.VideoCapture(0)  # to access the camera
label = ['closed','open']
closeCounter = 0
while (True):
    ret, frame = cap.read()
    #print(type(frame))
    frame = cv2.resize(frame, (640, 480))  # width,height
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_image, threshold1=30, threshold2=100)
    resized_image = cv2.resize(edge_img, (IMG_SIZE, IMG_SIZE))
    pred = np.argmax(model.predict([resized_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)]))
    
    if pred == 0:
        closeCounter += 1
        if closeCounter > 20:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        closeCounter = 0
        cv2.putText(frame, "Normal (open)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame with the prediction
    cv2.imshow("Face recognition", frame)
    cv2.imshow("edge image", edge_img)    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()


