# Thanos
Gesture based assistant

# Instructions

Run the python file. You may have to install some libraries. 

It will first ask for your name. Then it will ask that you provide it with a set of images so that it may classify what your usual background looks like.

After, it will ask for you to provide it with a gesture to classify. This gesture will be used to interact with the screenshot function. I used a fist.

Once it has collected all the neccesary data using OpenCv it will train them using a CNN (Convolutional Neural Network).

After all the 'calibration' has been done it will start recording you and will stake a screenshot when you hold up the gesture. The model is quite basic right now and as such requires that you hold the gesture for about 5 - 10 seconds. 

# Future

There is a lot to be done with the project.

The first step will be to optomize the data collection. We want to be able to record only the arm/hand for the gesture (Although possible facial features could be used in the future). Once we have this we can make much more accurate predictions. 

I would also like to optomize the neural network to make it lighter and more powerful. Using a pre classified model such as ResNet50 might prove to be helpful.

After this has been done I want to package it or make it into a usable product.

Feel free to contribute! 

Credit for the code goes to this article. Most of the code is from here: https://medium.com/swlh/hand-gestures-using-webcam-and-cnn-convoluted-neural-network-b02c47b3d5ab
