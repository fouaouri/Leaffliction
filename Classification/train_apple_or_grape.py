import os
from sklearn.model_selection import train_test_split #pip install scikit-learn
import numpy as np
import pandas as pd
import shutil
#The shutil module in Python is a utility library for file operations. Its name comes from “shell utilities.” You can use it to copy, move, or delete files and directories, among other things.
import subprocess
import json


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_generator = datagen.flow_from_directory(
    "Apple",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
print(train_generator.class_indices)

with open("class_grape_or_apple.json", "w") as f:
    json.dump(train_generator.class_indices, f)

validation_generator = datagen.flow_from_directory(
    "Apple",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Data generators created.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
num_classes = len(train_generator.class_indices)

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
    Dense(num_classes, activation='softmax')
])

print("Model architecture defined.")
#Compile & train the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled. Starting training...")
trained_model = model.fit(train_generator, epochs=20, validation_data=validation_generator)
print("Model training completed.")
# Save the model
model.save("apple_or_Grape_model.h5")