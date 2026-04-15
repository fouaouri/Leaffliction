import sys
import os
import cv2
import numpy as np

ROTATION_ANGLE = 20
BLUR_KERNEL = (7, 7)
ILLUMINATION_ALPHA = 1.0
ILLUMINATION_BETA = 40
CONTRAST_ALPHA = 1.4
CONTRAST_BETA = 0
SKEW_FACTOR_X = 0.1
SKEW_FACTOR_Y = 0.1
PERSPECTIVE_OFFSET = 0.05

def validate_arguments():
    if len(sys.argv) != 2 and len(sys.argv )!= 3:
        print("Usage: python Augmentation.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isfile(path):
        print("Error: file does not exist!")
        sys.exit(1)

    use_target_dir = False

    if len(sys.argv) == 3:
        if sys.argv[2] == "-t":
            use_target_dir = True
        else:
            print("Invalid flag. Use -t")
            sys.exit(1)

    return path , use_target_dir

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        print("Error: unable to read the image")
        sys.exit(1)

    return image

def extract_path_info(image_path):
    file_name = os.path.basename(image_path)
    class_name = os.path.basename(os.path.dirname(image_path))
    plant_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

    name, extension = os.path.splitext(file_name)

    return class_name, plant_name, name, extension

def create_output_directory(image_path, plant_name, class_name, use_target_dir):
    if use_target_dir:
        return os.path.dirname(image_path)
    else:
        output_dir = os.path.join("augmented_directory", plant_name, class_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

def save_image(output_dir, name, extension, aug_name, image):
    path = os.path.join(output_dir, f"{name}_{aug_name}{extension}")
    cv2.imwrite(path, image)
    print(f"Saved: {path}")

def flip_image(image):
    return cv2.flip(image, 1)

def rotate_image(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, ROTATION_ANGLE, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def blur_image(image):
    return cv2.GaussianBlur(image, BLUR_KERNEL, 0)

def illuminate_image(image):
    return cv2.convertScaleAbs(image, alpha=ILLUMINATION_ALPHA, beta=ILLUMINATION_BETA)

def contrast_image(image):
    return cv2.convertScaleAbs(image, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)

def skew_image(image):
    h, w = image.shape[:2]

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1]
    ])

    dst = np.float32([
        [w * SKEW_FACTOR_X, 0],
        [w - 1, h * SKEW_FACTOR_Y],
        [w * (SKEW_FACTOR_X * 2), h - 1]
    ])

    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, (w, h))

def projective_image(image):
    h, w = image.shape[:2]

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    offset = PERSPECTIVE_OFFSET

    dst = np.float32([
        [w * offset, h * offset],
        [w * (1 - offset), 0],
        [0, h - 1],
        [w * (1 - offset * 2), h * (1 - offset)]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h))

def apply_augmentations(image):
    return {
        "Flip": flip_image(image),
        "Rotate": rotate_image(image),
        "Blur": blur_image(image),
        "Illumination": illuminate_image(image),
        "Contrast": contrast_image(image),
        "Skew": skew_image(image),
        "Projective": projective_image(image),
    }

def main():
    image_path, use_target_dir = validate_arguments()
    image = load_image(image_path)

    class_name, plant_name, name, extension = extract_path_info(image_path)

    output_dir = create_output_directory(
        image_path,
        plant_name,
        class_name,
        use_target_dir
    )

    print("Image loaded successfully")
    print(f"Plant: {plant_name}")
    print(f"Class: {class_name}")

    if use_target_dir:
        print("Mode: SAVE IN ORIGINAL DIRECTORY (-t)")
    else:
        print("Mode: SAVE IN augmented_directory")

    augmentations = apply_augmentations(image)

    for aug_name, aug_image in augmentations.items():
        save_image(output_dir, name, extension, aug_name, aug_image)

if __name__ == "__main__":
    main()