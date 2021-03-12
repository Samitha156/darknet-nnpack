from picamera import PiCamera
from subprocess import Popen, PIPE
import threading
from time import sleep
import os, fcntl
import cv2

#NEW IMPORTS
from Motor_Control import Motorize
import numpy as np



#END NEW IMPORTS



####### NEW METHODS

def build_cnn():
    in_layer = Input(shape=(146, 44, 1))
    x = BatchNormalization()(in_layer)
    x = Conv2D(128, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 16 maps
    x = Conv2D(64, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 32 maps
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 64 maps
    x = Dropout(0.5)(x)
    x = Conv2D(128, (1, 1))(x)  # finally 128 maps for global average-pool
    x = Flatten()(x) # pseudo-dense 128 layer
    output_layer = Dense(3, activation="softmax")(x)  # softmax output
    model = Model(inputs=in_layer, outputs=output_layer)
    learning_rate = 1e-3
    optm = Adam(lr=learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #model.load_weights("alex_weights.h5")
    return model



######## END NEW METHOS


iframe = 0

camera = PiCamera()

#Yolo v3 is a full convolutional model. It does not care the size of input image, as long as h and w are multiplication of 32

#camera.resolution = (160,160)
camera.resolution = (416, 416)
#camera.resolution = (544, 544)
#camera.resolution = (608, 608)
#camera.resolution = (608, 288)


camera.capture('frame.jpg')
sleep(0.1)

#spawn darknet process
yolo_proc = Popen(["./darknet",
                   "detect",
                   "./cfg/yolov3-tiny.cfg",
                   "./yolov3-tiny.weights",
                   "-thresh","0.1"],
                   stdin = PIPE, stdout = PIPE)

fcntl.fcntl(yolo_proc.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
print("building ALEXNET")

print ("BUILT")
print("WORK")
'''while True:
    try:
        stdout = yolo_proc.stdout.read()
        if 'Enter Image Path' in stdout:
            try:
               im = cv2.imread('predictions.png')
               print(im.shape)
               cv2.imshow('yolov3-tiny',im)
               key = cv2.waitKey(5) 
            except Exception:
               pass
            camera.capture('frame.jpg')
            yolo_proc.stdin.write('frame.jpg\n')
        if len(stdout.strip())>0:
            print('get %s' % stdout)
    except Exception:
        pass
'''