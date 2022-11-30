import numpy as np
import imutils
import sys
import cv2
import os
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential 

images = []
image_labels  = []
directory = 'dataset'
list_of_files = os.listdir(directory)
index = 0
for file in list_of_files:
    subfiles = os.listdir(directory+'/'+file)
    for sub in subfiles:
        path = directory+'/'+file+'/'+sub
        img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(32,32,3)
        images.append(im2arr)
        image_labels.append(file)
    print(file)    

X = np.asarray(images)
Y = np.asarray(image_labels)
Y = to_categorical(Y)
img = X[20].reshape(32,32,3)
cv2.imshow('ff',cv2.resize(img,(250,250)))
cv2.waitKey(0)
print("shape == "+str(X.shape))
print("shape == "+str(Y.shape))
print(Y)
X = X.astype('float32')
X = X/255

np.save("img_data.txt",X)
np.save("img_label.txt",Y)

X = np.load('img_data.txt.npy')
Y = np.load('img_label.txt.npy')
print(Y)
img = X[20].reshape(32,32,3)
cv2.imshow('ff',cv2.resize(img,(250,250)))
cv2.waitKey(0)

classifier = Sequential() #alexnet transfer learning code here
classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X, Y, batch_size=32, epochs=50)
classifier.save_weights('model/train.h5')            
model_json = classifier.to_json()
with open("model/train.json", "w") as json_file:
    json_file.write(model_json)
print(classifier.summary())



