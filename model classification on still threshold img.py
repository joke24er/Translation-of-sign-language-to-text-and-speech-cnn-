import cv2
import tensorflow as tf
import numpy as np
from cam import trans

CATEGORIES = ['aboard','all_gone','baby','beside','book','bowl','bridge','camp','cartridge',
              'eight','five','fond','four','friend','glove','hang','high','house','how_many',
              'loeMe','man','marry','meat','medal','opposite']

z=0
key = ord('e')
IMG_SIZE=200
img_array=cv2.imread('data2/test/glove/glove.961.jpg')
img_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

lower_red = np.array([20, 100, 100])
upper_red = np.array([30, 255, 255])

mask = cv2.inRange(img_hsv, lower_red, upper_red)

res = cv2.bitwise_and(img_array, img_array, mask=mask)
#color = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
color = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
new_array=cv2.resize(color,(200,200))
#k=new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
k=new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)
model=tf.keras.models.load_model("F:\\datasets\\exp (3).model")
#model=tf.keras.models.load_model("F:\\datasets\\gray1.model")
prediction=model.predict([k])
for i in range(0,25):
    if (prediction[0][i]==1):
        z=i
        trans('hi', CATEGORIES[z])


print(CATEGORIES[z])
