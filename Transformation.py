import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_original(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image.copy()
    plt.subplot(3, 2, 1)
    plt.imshow(output)
    plt.title("Figure IV.1: Original", y=-0.2)
    plt.axis('off')
    return output

def show_gassian_blur(image):
    output = image.copy()
    grey = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    n, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.subplot(3, 2, 2)
    plt.imshow(thresholded, cmap='gray')
    plt.title("Figure IV.2: Gaussian Blur", y=-0.2)
    plt.axis('off')
    return output

def show_mask(image):
    output = image.copy()
    hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    background = np.ones_like(output) * 255
    mask =cv2.inRange(hsv, lower_green, upper_green)
    result = np.where(mask[:, :, None] == 255, output, background)

    plt.subplot(3, 2, 3)
    plt.imshow(result)
    plt.title("Figure IV.3: Mask", y=-0.2)
    plt.axis('off')
    return mask, output
    
def show_roi_objects(image, mask):
    output = image.copy()
    roi = cv2.bitwise_and(output, output, mask=mask)
    output[mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(output, 0.1, output, 0.8, 0)
    plt.subplot(3, 2, 4)
    plt.imshow(blended)
    plt.title("Figure IV.4: ROI Objects", y=-0.2)
    plt.axis('off')
    return output

def show_analyse_objects(image, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.Canny(mask, 100, 200)
    output = image.copy()

    for contour in contours:
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 4)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 255), 2)
    output[edges > 0] = [0, 0, 255]
    plt.subplot(3, 2, 5)
    plt.imshow(output)
    plt.title("Figure IV.5: Analyse Objects", y=-0.2)
    plt.axis('off')
    return output

def showpsodolandmarks(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()

    for contour in contours:
        step = max(1, len(contour) // 30)
        for i in range(0, len(contour), step):
            x, y = contour[i][0]
            cv2.circle(output_image, (x, y), 4, (255, 0, 255), -1)

    plt.subplot(3, 2, 6)
    plt.imshow(output_image)
    plt.title("Figure IV.6: Pseudo Landmarks", y=-0.2)
    plt.axis('off')
    return output_image

def main ():
    if(len(sys.argv) == 2):
        imagePath = sys.argv[1]
        image = cv2.imread(imagePath)
        show_original(image)
        show_gassian_blur(image)
        mask, output = show_mask(image)
        show_roi_objects(image, mask)
        show_analyse_objects(image, mask)
        showpsodolandmarks(image, mask)
        plt.show()
    elif(len(sys.argv) == 6):
        dst = sys.argv[4]
        os.makedirs(dst, exist_ok=True)
        src = sys.argv[2]
        for file in os.listdir(src):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(src, file)
                name, extension = os.path.splitext(file)
                image = cv2.imread(path)
                plt.figure(figsize=(10, 12))

                show_original(image)
                show_gassian_blur(image)
                mask, mask_image = show_mask(image)
                show_roi_objects(image, mask)
                show_analyse_objects(image, mask)
                showpsodolandmarks(image, mask)

                plt.savefig(os.path.join(dst, f"{name}_transformed{extension}"), dpi=300)
                plt.close()

    else:
        print("Usage: python Transformation.py <filename>")
        return

main()