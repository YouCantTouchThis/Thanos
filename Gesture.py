#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
import os
from cv2 import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import time
import pyautogui


def dataGen(path, imagecount):
    directory = path
    imagecount = imagecount

    os.makedirs(directory, exist_ok=True)

    video = cv2.VideoCapture(0)

    filename = len(os.listdir(directory))
    count = 0
    pbar = tqdm(total = imagecount+1)
    while True and count < imagecount:

        filename += 1
        count += 1
        _, frame = video.read()
        kernel = np.ones((3,3),np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



        # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)

        #extract skin colur image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)



        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)

        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100)
        path = directory+"//"+str(filename)+".jpg"
        cv2.imwrite(path, mask)
        cv2.imshow("Capturing", mask)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    pbar.update(1)
    pbar.close()
    video.release()
    cv2.destroyAllWindows()


# In[14]:


from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

def trainer():
    Imagesize=128
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(Imagesize,Imagesize,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.255)

    train_generator = train_datagen.flow_from_directory('~/OPENDATA/train',
    target_size=(Imagesize,Imagesize),batch_size=64, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory('~/OPENDATA/test', target_size=(Imagesize,Imagesize), batch_size=64, class_mode='categorical')

    model.fit_generator(train_generator, epochs=5, steps_per_epoch=63, validation_data=validation_generator, validation_steps=7, workers=4)
    model.save("gesture.h5")


# In[1]:


import cv2
import numpy as np
from keras import models
import sys
from PIL import Image
from twilio.rest import Client

counter = 0
def executer():
    video_capture = cv2.VideoCapture(0)

    #Load the saved model
    model = models.load_model('gesture.h5')
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        kernel = np.ones((3,3),np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



        # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)

        #extract skin colur image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)



        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)

        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100)
        mask = cv2.resize(mask,(128,128))
        img_array = np.array(mask)
        #print(img_array.shape)
        img_array = np.stack((img_array,)*3, axis=-1)
        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array_ex = np.expand_dims(img_array, axis=0)
        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array_ex)
        global counter
        if(prediction[0][0] == 0):
            counter += 1
        else:
            counter = 0
        if(counter == 25):
            pyautogui.screenshot("~/ThanosScreenshot.png")
            print("Screenshot Taken")
            break
        print(prediction[0][0])
        print(counter)
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


# In[29]:


# Lets make this clean

print("Welcome to Thanos where everything is just a snap away!")
print("Lets get started with some setup first...")
name = ""
while(len(name) <= 0):
    name = input("Name: ")
print("Thanks!")

print("Lets do some gesture setup, first we will need to store your base sorroundings!")
time.sleep(2)
dataGen("~/OPENDATA/train/0", 1000)
dataGen("~/OPENDATA/test/0", 200)

yes = input("Are you ready to continue? (y/n)")
if(yes == "y"):
    print("Now we need you to make a custom gesture. This well help us take screenshots of your screen!")
    print("Please use the same gesture the whole time and give us different angles for maximum calibration!")
    time.sleep(2)

    dataGen("~/OPENDATA/train/1", 1000)
    dataGen("~/OPENDATA/test/1", 200)

    print("Give us a second to process everything!")
    trainer()
    executer()

