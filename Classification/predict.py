
import sys
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
def show_prediction_graph(classes, pred):
    plt.bar(classes, pred)
    plt.ylabel("probability")
    plt.title("Model prediction")
    plt.show()

model = load_model("apple_leafs_model.h5")

with open("class_indices_apple.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    img = preprocess_image(img_path)
    
    prediction = model.predict(img)
    
    print("Raw prediction:", prediction)
    
    predicted_class = class_names[np.argmax(prediction)]
    
    print("Predicted class:", predicted_class)
    show_prediction_graph(class_names, prediction[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    predict(sys.argv[1])