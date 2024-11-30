import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load preprocessed data arrays
xTrain = np.load("xTrain.npy")
xTest = np.load("xTest.npy")
yTrain = np.load("yTrain.npy")
yTest = np.load("yTest.npy")

# Define constants for the model
inputShape = (128, 128, 3)
numClasses = 8

# Set up data augmentation for training images
dataAugmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to create the baseline CNN model
def buildCnnModel(inputShape, numClasses):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(numClasses, activation='softmax')
    ])
    return model

# Function to create the ResNet50 model with transfer learning
def buildResNetModel(inputShape, numClasses):
    baseModel = applications.ResNet50(weights='imagenet', include_top=False, input_shape=inputShape)
    baseModel.trainable = False  # Freeze the base model layers
    
    model = models.Sequential([
        baseModel,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(numClasses, activation='softmax')
    ])
    return model

# Instantiate both models
cnnModel = buildCnnModel(inputShape, numClasses)
resNetModel = buildResNetModel(inputShape, numClasses)

# Function to compile and train a model
def compileAndTrainModel(model, xTrain, yTrain, xVal, yVal, batchSize=32, epochs=10):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        dataAugmentation.flow(xTrain, yTrain, batch_size=batchSize),
        validation_data=(xVal, yVal),
        epochs=epochs
    )
        # Display final training and validation metrics
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    return history

# Train the CNN model
print("Training the baseline CNN model...")
cnnHistory = compileAndTrainModel(cnnModel, xTrain, yTrain, xTest, yTest, epochs=20)

# Train the ResNet50 model
print("Training the ResNet50 model with transfer learning...")
resNetHistory = compileAndTrainModel(resNetModel, xTrain, yTrain, xTest, yTest, epochs=10)

# Evaluate both models on the test set
def evaluateModel(model, xTest, yTest, modelName):
    testLoss, testAccuracy = model.evaluate(xTest, yTest)
    print(f"{modelName} - Test Loss: {testLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")

print("\nEvaluating CNN Model:")
evaluateModel(cnnModel, xTest, yTest, "CNN Model")

print("\nEvaluating ResNet50 Model:")
evaluateModel(resNetModel, xTest, yTest, "ResNet50 Model")

# Function to plot training and validation accuracy/loss
def plotTrainingHistory(history, modelName):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{modelName} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{modelName} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot training history for both models
print("\nPlotting training history for CNN model...")
plotTrainingHistory(cnnHistory, "CNN Model")

print("\nPlotting training history for ResNet50 model...")
plotTrainingHistory(resNetHistory, "ResNet50 Model")
