# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
import cv2
from matplotlib import pyplot
from random import randrange
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Activation,BatchNormalization
from keras.models import model_from_json
import numpy as np
import cv2
from imutils import paths
import imutils
import matplotlib.pyplot as plt
from keras.preprocessing import image

# load model
'''
arr = ['010','020','030','040','050','060','070','080','090','100']
for i in range(len(arr)):
    model = load_model('model/generator_model_'+arr[i]+'.h5')
    vector = asarray([[0.75 for _ in range(200)]])
    X = model.predict(vector)
    img = X[0, :, :]
    cv2.imshow(arr[i],cv2.resize(img,(250,250)))
    cv2.waitKey(0)
'''
with open('model/train.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model/train.h5")
loaded_model._make_predict_function()   
print(loaded_model.summary())
'''
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    print(x_input.shape)
    return x_input

arr = ['010','020','030','040','050','060','070','080','090','100']
for i in range(len(arr)):
    model = load_model('model/generator_model_'+arr[i]+'.h5')
    latent_points = generate_latent_points(200, 200)
    X = model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    #X = (X + 1) / 2.
    #create_plot(X, 10)
    index = randrange(200)
    print(index)
    img = X[index, :, :]
    #img = cv2.resize(img, (32,32))
    #img 
    #im2arr = np.array(img)
    #im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(img)
    img = X.astype('float32')
    img = X/255
    preds = loaded_model.predict(X)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    print(predict)
    #img = im2arr.reshape(64,64,3)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(str(predict),cv2.resize(X[index, :, :],(250,250)))
    cv2.waitKey(0)
'''

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

arr = ['010','020','030','040','050','060','070','080','090','100']
for i in range(len(arr)):
    model = load_model('model/generator_model_080.h5')
    latent_points = generate_latent_points(200, 200)
    X = model.predict(latent_points)
    index = randrange(200)
    print(index)
    img = X[index, :, :]
    img1 = np.asarray(img)
    img1 = img1.reshape(1,32,32,3)
    #img1 = img1.astype('float32')
    #img1 = img1/255
    preds = loaded_model.predict(img1)
    predict = np.argmax(preds)
    print(preds)
    cv2.imshow(str(predict),cv2.resize(img,(250,250)))
    cv2.waitKey(0)








