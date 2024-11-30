# Natural Image Classification with Deep Learning

This project demonstrates image classification using deep learning models on a dataset of natural images, each labeled into one of eight categories: Airplane, Car, Cat, Dog, Flower, Fruit, Motorbike, and Person. Two models are built and evaluated:

1.  A simple Convolutional Neural Network (CNN) as a baseline.
    
2.  A ResNet50 model using transfer learning to achieve higher accuracy.
    

## Table of Contents

-   Project Overview
    
-   Dataset Information
    
-   Setup and Installation
    
-   Project Structure
    
-   Steps to Run the Project
    

-   1. Data Preparation
    
-   2. Model Training and Evaluation
    

-   Expected Outputs
    
-   Troubleshooting and Tips
    
-   License
    

----------

## Project Overview

This project explores deep learning approaches for image classification. A dataset of natural images is used, covering eight distinct classes. The objective is to accurately classify these images by leveraging two types of neural network models:

-   Baseline CNN Model: A simple convolutional network that provides a foundational accuracy level for comparison.
    
-   ResNet50 Transfer Learning Model: A pre-trained ResNet50 model fine-tuned to classify the natural images dataset, expected to achieve superior accuracy due to its advanced architecture and transfer learning from large-scale image data.
    

## Dataset Information

Dataset Source: The dataset is available on [Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images). You must download and extract it before running the project.

Dataset Structure: The dataset contains eight folders (one for each class) inside a main folder named natural_images. Each folder contains images of the respective category:

-   Folder names: airplane, car, cat, dog, flower, fruit, motorbike, person
    
-   Images are of varying dimensions, so they will be resized for uniformity before training.
    

After downloading, ensure the folder structure looks like this:

  

natural_images/

├── airplane/

├── car/

├── cat/

├── dog/

├── flower/

├── fruit/

├── motorbike/

└── person/

  


## *Note*: Update the `dataDir` variable in line 7 of the `prepareData.py` file to the appropriate path to the `natural_images` folder.

## Setup and Installation

### Requirements

-   Python 3.7+ is required.
    
-   Libraries:
    

-   TensorFlow (for deep learning models)
    
-   Keras (integrated with TensorFlow for model building)
    
-   Numpy (for handling arrays)
    
-   Matplotlib (for plotting results)
    

Install all required libraries by running:

pip install numpy tensorflow matplotlib

  

### Hardware Recommendation

While the project can run on a CPU, using a GPU (such as on Google Colab or a local GPU setup) will significantly speed up training, especially for the ResNet50 model.

## Project Structure

1.  prepareData.py: A script to load, preprocess, and split the dataset into training and test sets. It outputs four .npy files (xTrain.npy, xTest.npy, yTrain.npy, yTest.npy) for use in model training.
    
2.  trainModel.py: A script to load the preprocessed data, build the CNN and ResNet50 models, train them, and evaluate their performance on the test set.
    

## Steps to Run the Project

### 1. Data Preparation

This step prepares the dataset by:

-   Loading images from each class folder.
    
-   Resizing images to a uniform size (128x128 pixels).
    
-   Normalizing pixel values to a range of 0 to 1.
    
-   Splitting data into 80% training and 20% test sets.
    
-   Saving these processed arrays as .npy files for efficient loading during model training.
    

#### To Run Data Preparation

1.  Open a terminal in the project directory.
    

Run the following command to execute prepareData.py:  
  
python prepareData.py

2.    
    

#### Expected Output of Data Preparation

Upon successful execution, you will see a message confirming the completion of data preparation. Four files will be saved:

-   xTrain.npy: Training set images.
    
-   xTest.npy: Test set images.
    
-   yTrain.npy: Labels for the training set.
    
-   yTest.npy: Labels for the test set.
    

### 2. Model Training and Evaluation

This step involves:

-   Loading the Preprocessed Data: The .npy files created in the previous step are loaded for model training.
    
-   Building the Models:
    

-   A simple CNN model is built to serve as a baseline.
    
-   A ResNet50 model with frozen pre-trained layers is created to leverage transfer learning.
    

-   Training: Both models are trained using the Adam optimizer and sparse categorical cross-entropy loss. The CNN model is trained for 20 epochs, while the ResNet50 model is trained for 10 epochs.
    
-   Evaluation: Both models are evaluated on the test set to compare performance. The results include accuracy and loss metrics for each model.
    

#### To Run Model Training and Evaluation

1.  Ensure prepareData.py has been successfully executed.
    

Run the following command to execute trainModel.py:  
  
python trainModel.py

2.    
    

#### Expected Outputs of Model Training and Evaluation

-   Training and validation accuracy/loss metrics for each epoch will be printed for both models.
    
-   Final test accuracy and loss for each model will be displayed.
    
-   Additionally, accuracy and loss plots for both training and validation sets are generated for visual analysis.
    

## Expected Outputs

-   Baseline CNN Model:
    

-   Expected to achieve moderate accuracy, acting as a benchmark for comparison.
    
-   Accuracy and loss metrics are printed after training and testing.
    

-   ResNet50 Transfer Learning Model:
    

-   Expected to achieve higher accuracy due to the strength of transfer learning.
    
-   Accuracy and loss metrics for both training and test data are printed and plotted.
    

## Troubleshooting and Tips

-   File Not Found Error: Ensure that the .npy files generated by prepareData.py are in the same directory as trainModel.py. Rerun prepareData.py if they are missing.
    

Module Not Found: If TensorFlow, Keras, or Matplotlib are not found, verify that you have installed all dependencies by running:  
  
pip install numpy tensorflow matplotlib

-     
    
-   Memory Issues: Training on large datasets can consume significant memory. Use a GPU if available, and close other applications to free up system resources.
    
-   Runtime on CPU: Running ResNet50 training on a CPU can be very slow. If you don’t have a GPU available, consider reducing the number of epochs or using a smaller dataset subset for faster results.
    

## License

This project is for educational purposes only and follows standard fair use for datasets.
