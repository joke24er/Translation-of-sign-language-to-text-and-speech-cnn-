import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import models
from cam import trans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load the saved model

video = cv2.VideoCapture(0)
IMG_SIZE=200
#model=tf.keras.models.load_model("F:\datasets\\naya (2) (2).model")
model=tf.keras.models.load_model("F:\datasets\\exp (3).model")
#model=tf.keras.models.load_model("F:\datasets\\yellow1.model")
CATEGORIES = ['aboard','all_gone','baby','beside','book','bowl','bridge','camp','cartridge',
              'eight','five','fond','four','friend','glove','hang','high','house','how_many',
              'loeMe','man','marry','meat','medal','opposite']
z=0

while True:
        _, frame = video.read()

        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([20, 100, 100])
        upper_red = np.array([30, 255, 255])

        mask = cv2.inRange(img_hsv, lower_red, upper_red)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        img_array = cv2.resize(res, (IMG_SIZE, IMG_SIZE))
        new_array = img_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        prediction = model.predict([new_array])
        for i in range(0, 25):
            if (prediction[0][i] == 1):
                z = i

        print(CATEGORIES[z])
        cv2.putText(res, CATEGORIES[z], (10, 355 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 1)

        cv2.imshow("Capturing", res)
        key = cv2.waitKey(1)
        if key == ord('h'):
            trans('hi',CATEGORIES[z])
        if key == ord('t'):
            trans('ta',CATEGORIES[z])


        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()