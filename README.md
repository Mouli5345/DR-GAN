# DR-GAN

## Introduction

Diabetic retinopathy is a devastating disorder of the eyes that is linked to diabetes (DR). Based on international convention, a severity scale with five levels can be used. However, a sizable amount of balanced training data must be acquired in order to optimise a grading model for strong generalizability, and this is particularly difficult for the high severity levels. Data with a considerable amount of variability cannot be produced by common data augmentation methods like random flipping and rotation. This programme was created to detect diabetic retinopathy in patients' fundus photographs. This software promises to make it easier for diabetic individuals to identify their retinal issues and seek medical attention more quickly. The model that will be used to identify diabetic retinopathy in patients will be trained using the GAN technique.

## GAN model
It stands for Generative Adversarial Network. It is utilised in unsupervised learning-based machine learning tasks. It comprises of two models that automatically identify and evaluate patterns in data input. The terms "Generator" and "Discriminator" are used to describe the two models. They engage in competition with one another to identify, catalogue, and replicate the differences that exist in a dataset. GANs can be used to create new instances that the original dataset could have created.

## Model Performance
For the Discriminator to train on and categorise the images produced by the Generator model, we used the [**FGADR dataset**](https://csyizhou.github.io/FGADR/) as a reference. The discriminator effectively categorises the images with an average grading accuracy of 80%.

## Running the project

1. Download the prerequisite packages
    - Run the command ```pip install -r requirements.txt``` on your terminal.

2. Run the driver program
    - Run the ```run.bat``` file.       
    **OR**
    - Run the command ```python Main.py``` in your project's root directory

3. Upload dataset
    - Upload the Fundus dataset.
    - This will be used to generate images by the GAN model

4. Load GAN model
    - Click on Load GAN model to generate images

5. Load Diabetic Retinopathy Prediction model
    - Now load the Diabetic Retinopathy Prediction model by clicking on the button
    - This is a CNN model traned to classify the pictures into one of the five categories of Diabetic Retinopathy categories available in the dataset.

6. Make predictions on GAN images
    - Now make predictions on the severity category of the  images produced by the GAN model using the CNN model. 
    - Perform this by clicking on Generate GAN Image & Predict Severity.
