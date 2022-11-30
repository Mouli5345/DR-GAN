
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from keras.models import model_from_json
from random import randrange
from numpy.random import randn
import cv2
from keras.models import load_model
from matplotlib import pyplot

main = tkinter.Tk()
main.title("Diabetic Retinopathy Image Synthesis with Generative Adversarial Network") #designing main screen
main.geometry("1300x1200")

global filename
global gan_model
global predict_model

def upload():
    text.delete('1.0', END)
    global filename
    global X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    print(x_input.shape)
    return x_input

def create_plot(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :])
    pyplot.show()

def ganModel():
    global gan_model
    gan_model = load_model('model/generator_model_080.h5')
    latent_points = generate_latent_points(200, 200)
    X = gan_model.predict(latent_points)
    text.insert(END,'GAN model generated\n')
    text.insert(END,'GAN generated new images size : '+str(X.shape)+"\n\n")
    create_plot(X, 10)

def predictModel():
    global predict_model
    text.delete('1.0', END)
    with open('model/train.json', "r") as json_file:
        loaded_model_json = json_file.read()
        predict_model = model_from_json(loaded_model_json)

    predict_model.load_weights("model/train.h5")
    predict_model._make_predict_function()
    print(predict_model.summary())
    text.insert(END,'See black console to view model summary')

def getPrediction(img):
    result = 'none'
    img1 = np.asarray(img)
    img1 = img1.reshape(1,32,32,3)
    preds = predict_model.predict(img1) #predicting class of image severity
    predict = np.argmax(preds) #get then class value
    result = 'none'
    if predict == 0:   #if value 0 then result is NO DR
        result = 'No DR'
    if predict == 1:   #if value 1 mean result is MILD
        result = 'Mild'
    if predict == 2:   #if value 2 mean result is Moderate
        result = 'Moderate'
    if predict == 3:   
        result = 'Severe'
    if predict == 4:   #if value 4 mean result is Proliferative DR
        result = 'Proliferative DR'
    return result    
    

def predictSeverity():
    latent_points = generate_latent_points(200, 200) #making array of 200 to ask GAN to generate 200 images from train model
    X = gan_model.predict(latent_points) #calling GAN predict model with 200 array size to generate image
    for i in range(0,10):   #displying 200 images prediction will be difficult so randomly choosing 10 images out of 200 gan images
        index = randrange(200) #randomly generating index 
        img = X[index, :, :] #reading GAN image using random index
        result = getPrediction(img)# calling get prediction method to predict severity of generated image
        img = cv2.resize(img,(300,300)) #resizeing image
        cv2.putText(img, 'Prediction Result : '+result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)#displaying severity result on image
        cv2.imshow('image id : '+str(index)+' Prediction Result : '+result,img)#severity result with image id
    cv2.waitKey(0)
    
    
def closeApp():
    main.destroy()
    

font = ('times', 16, 'bold')
title = Label(main, text='Diabetic Retinopathy Image Synthesis with Generative Adversarial Network')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Fundus Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

ganButton = Button(main, text="Load GAN Model", command=ganModel)
ganButton.place(x=50,y=150)
ganButton.config(font=font1) 

modelButton = Button(main, text="Load Diabetic Retinopathy Prediction Model", command=predictModel)
modelButton.place(x=50,y=200)
modelButton.config(font=font1) 

predictButton = Button(main, text="Generate GAN Image & Predict Severity", command=predictSeverity)
predictButton.place(x=50,y=250)
predictButton.config(font=font1) 

closeButton = Button(main, text="Exit", command=closeApp)
closeButton.place(x=50,y=300)
closeButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
