#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:24:03 2020

@author: shoumik
"""

# Import few more necessary libraries.
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Function to load and prepare the image in right shape
def load_image(filename):
	# Load the image
    img = load_img(filename, target_size=(224, 224))
    	# Convert the image to array
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Load an image and predict the apparel class
filepath='/home/shoumik/Desktop/face-mask-detector shoumik/Kriti-Kharbanda.jpg'
img = load_image(filepath)
# Load the saved model
model = load_model('Mask_detector_model.h5')
# Predict the apparel class
prediction = model.predict(img)

#loading the cascades
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
image = cv2.imread(filepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    
    if(prediction[0][0]>0.5):
        pred='Mask'
        #print(pred)
        cv2.putText(image,pred,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        pred='No Mask'
        #print(pred)
        cv2.putText(image,pred,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0) & 0xFF==ord('q')
#Map apparel category with the numerical class
'''if (prediction[0][0]>0.5):
  pred = "Mask"
else:
  pred = "No Mask"'''
  
print(pred)


cv2.destroyAllWindows()