import os
from sklearn.model_selection import train_test_split #pip install scikit-learn
import numpy as np
import pandas as pd
import shutil
#The shutil module in Python is a utility library for file operations. Its name comes from “shell utilities.” You can use it to copy, move, or delete files and directories, among other things.
import subprocess
import json

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
    train_images, validation_images = train_test_split(images, test_size=0.1, random_state=42)
    trainClass = os.path.join(train_dir, className)
    validationClass = os.path.join(validation_dir, className)
    
    os.makedirs(os.path.join(train_dir, className), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, className), exist_ok=True)

    for image in train_images:
        shutil.copy(os.path.join(classPath, image), os.path.join(train_dir, className, image))
        # print(f"Copied to train: {image}")
    for image in validation_images:
        shutil.copy(os.path.join(classPath, image), os.path.join(validation_dir, className, image))
        # print(f"Copied to validation {image}")

# print("Data has been split into training and validation sets.")


# run the Augmentation program

for className in os.listdir(train_dir):
    classPath = os.path.join(train_dir, className)
    if not os.path.isdir(classPath):
        continue
    for image in os.listdir(classPath):
        if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        imagePath = os.path.join(classPath, image)
        subprocess.run(["python3", "../Augmentation.py", imagePath])

print("Data augmentation completed.")

# prepare data for CNN model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 0-255 => 0-1

train_data_generator = ImageDataGenerator(rescale=1./255)
validation_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
print(train_generator.class_indices)

with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
validation_generator = validation_data_generator.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
print("Data generators created.")
#CNN model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(128,128,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

print("Model architecture defined.")
#Compile & train the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled. Starting training...")
trained_model = model.fit(train_generator, epochs=20, validation_data=validation_generator)
print("Model training completed.")

# Save the model
model.save("grape_leafs_model.h5")