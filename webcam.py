#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:04:06 2020

@author: shoumik
"""


import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
#load the model
model=load_model('Mask_detector_model.h5')

#loading the cascades
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#webcam face recognition
def cam_stream():
    video_capture=cv2.VideoCapture(0)
    while True:
        _,frame=video_capture.read()
        faces=face_cascade.detectMultiScale(frame,1.3,5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            face=frame[y:y+h,x:x+w]
            cropped_face=face
        
            if type(face) is np.ndarray:
                face=cv2.resize(face,(224,224))
                im=Image.fromarray(face,'RGB')
                img_array=np.array(im)
                img_array=np.expand_dims(img_array,axis=0)
                global sess
                global graph
                with graph.as_default():
                    set_session(sess)
                    pred= model.predict(img_array)
            #pred=model.predict(img_array)
                print(pred)
                
                if(pred[0][0]>0.5):
                    prediction='Mask'
                    cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                else:
                    prediction='No Mask'
                    cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame,'No Face Found',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            #return cv2.imencode('.jpg', frame)[1].tobytes()    
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    video_capture.release()

    cv2.destroyAllWindows()
