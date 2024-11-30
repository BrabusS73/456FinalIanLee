import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# Set path to dataset directory
dataDir = "C:/Users/wised/Downloads/archive/natural_images"
imageSize = (128, 128)  # Target image size
classNames = sorted(os.listdir(dataDir))  # Class folders in alphabetical order

# Initialize lists for images and labels
images = []
labels = []

# Load and preprocess images
for label, className in enumerate(classNames):
    classDir = os.path.join(dataDir, className)
    for imageName in os.listdir(classDir):
        imagePath = os.path.join(classDir, imageName)
        # Load image, resize, and convert to array
        image = load_img(imagePath, target_size=imageSize)
        imageArray = img_to_array(image) / 255.0  # Normalize
        images.append(imageArray)
        labels.append(label)  # Label based on folder

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split into train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save arrays to disk
np.save("xTrain.npy", xTrain)
np.save("xTest.npy", xTest)
np.save("yTrain.npy", yTrain)
np.save("yTest.npy", yTest)

print("Data preparation complete. Arrays saved as .npy files.")
