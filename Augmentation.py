import cv2
import os
import sys
import numpy as np



def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 Augmentation.py <filename>")
        return
    imagePath = sys.argv[1]
    image = cv2.imread(imagePath)
    if image is None:
        print("Could not read the image.")
        return
    directoryName = os.path.dirname(imagePath)
    os.makedirs(directoryName, exist_ok=True)
    filename = os.path.basename(imagePath)
    name, extention = os.path.splitext(filename)
    # cv2.imwrite(f"{directoryName}/{name}_original{extention}", image)
    #Flip
    flipedImage = cv2.flip(image, 1)
    cv2.imwrite(f"{directoryName}/{name}_Flip{extention}", flipedImage)

    #rotation
    (h, w) = shape = image.shape[:2]
    center = (w // 2, h // 2)
    matrix2D = cv2.getRotationMatrix2D(center, 25, 1.0)
    rotatedImage = cv2.warpAffine(image, matrix2D, (w, h))
    cv2.imwrite(f"{directoryName}/{name}_Rotate{extention}", rotatedImage)
    
    #blured

    bluredImage = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite(f"{directoryName}/{name}_Blured{extention}", bluredImage)

    #contrast
    alpha = 1.5
    beta = 0
    contrastImage = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imwrite(f"{directoryName}/{name}_Contrast{extention}", contrastImage)

    #scaling

    scaled = cv2.resize(image, None, fx=1.2, fy=1.2)
    scaledImage = cv2.resize(scaled, (w, h))
    cv2.imwrite(f"{directoryName}/{name}_Scaled{extention}", scaledImage)

    #rightness
    brightImage = cv2.convertScaleAbs(image, alpha=1, beta=50)
    cv2.imwrite(f"{directoryName}/{name}_Bright{extention}", brightImage)

    #projective
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, 0], [w, 0], [50, h], [w - 50, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    projectiveImage = cv2.warpPerspective(image, matrix, (w, h))
    cv2.imwrite(f"{directoryName}/{name}_Projective{extention}", projectiveImage)

    

main()
