from google.colab import drive
drive.mount('/content/drive')

# Here the model is trained for the edge images from my own dataset prepared and the whole face is considered for training
import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import keras
import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight
from keras.optimizers import SGD
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = tf.keras.models.load_model("/content/CNN_model_2.h5",compile=False)
def check_eye_state(frame):
  IMG_SIZE = 145
  # Load the pre-trained cascade classifiers for eyes
  face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_alt.xml')
  eye_cascade = cv2.CascadeClassifier('/content/haarcascade_eye.xml')

  # Load the facial landmark indices for the eyes
  (l_start, l_end) = (42, 47)  # Left eye landmarks indices
  (r_start, r_end) = (36, 41)  # Right eye landmarks indices

  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(frame, 1.3, 5)
  pred = []
  for (x, y, w, h) in faces:
    roi_face = frame[y:y + h, x:x + w]
    #roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_face)

    for (ex,ey,ew,eh) in eyes:
        roi_eye = roi_face[ey:ey + eh, ex:ex + ew]
        resized_image = cv2.resize(roi_eye, (IMG_SIZE, IMG_SIZE))
        temp = np.argmax(model.predict([resized_image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)]))
        pred.append(temp)

  if sum(pred) <=1:
    return 0
  else:
    return 1

def edge_get_data(dir_path="/content/drive/MyDrive/People eyes and face/"):
  labels = ['hemanth','nheha','sai', 'sandeep','saptarsi']
  IMG_SIZE = 200
  data = []
  for label in labels:
    path = os.path.join(dir_path, label, 'face').replace('\\','/')
    #class_num = labels.index(label)
    print("reading faces of " +label+"...")
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        class_num = check_eye_state(img_array)
        gray_image = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
        edge_img = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        resized_array = cv2.resize(edge_img, (IMG_SIZE, IMG_SIZE))
        data.append([resized_array, class_num])
      except Exception as e:
        print(e)
  return data

new_data = edge_get_data()
print(len(new_data))

np.array(X).shape

X = []
y = []
for features in new_data:
    #print(features[0], features[1])
    X.append(features[0])
    y.append(features[1])
#print(X)
X = np.array(X).reshape(-1, 200, 200, 1)

y = np.array(y)


# Split the dataset into training and testing sets
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Data augmentation
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30, fill_mode='nearest', shear_range=0.2)
test_generator = ImageDataGenerator(rescale=1/255,  zoom_range=0.2, horizontal_flip=True, rotation_range=30, fill_mode='nearest', shear_range=0.2)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

train_images, test_images = X_train / 255.0, X_test / 255.0
test_labels = y_test
train_labels = y_train
class_names = ['closed','open']

# model creation for eye open and closing detection
model = Sequential()
model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=train_images.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50,
                    validation_data=(test_images, test_labels))

plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

model.save("models/recognition_model_2.h5")
model.save("models/recognition.model")

