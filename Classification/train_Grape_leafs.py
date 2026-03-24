import os
from sklearn.model_selection import train_test_split #pip install scikit-learn
import numpy as np
import pandas as pd
import shutil
#The shutil module in Python is a utility library for file operations. Its name comes from “shell utilities.” You can use it to copy, move, or delete files and directories, among other things.

# Split the data into training and validation sets
directoryPath = "../leaves/Grape"
train_dir = "./Grape/train"
validation_dir = "./Grape/validation"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

for className in os.listdir(directoryPath):
    classPath = os.path.join(directoryPath, className)
    if not os.path.isdir(classPath):
        continue
    images = [f for f in os.listdir(classPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Class: {className}, Found images: {images}")
    train_images, validation_images = train_test_split(images, test_size=0.1, random_state=42)
    trainClass = os.path.join(train_dir, className)
    validationClass = os.path.join(validation_dir, className)
    
    os.makedirs(os.path.join(train_dir, className), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, className), exist_ok=True)

    for image in train_images:
        shutil.copy(os.path.join(classPath, image), os.path.join(train_dir, className, image))
        print(f"Copied to train: {image}")
    for image in validation_images:
        shutil.copy(os.path.join(classPath, image), os.path.join(validation_dir, className, image))
        print(f"Copied to validation {image}")

print("Data has been split into training and validation sets.")


