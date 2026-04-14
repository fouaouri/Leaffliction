import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================
# CONFIGURATION
# ==================================
FIGURE_SIZE = (12, 10)
BLUR_KERNEL = (7, 7)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

BLUE_COLOR = (255, 0, 0)
PINK_COLOR = (255, 0, 255)
GREEN_COLOR = (0, 255, 0)
ORANGE_COLOR = (255, 165, 0)

OUTPUT_ROOT = "transformation_output"


# ==================================
# VALIDATION / PATHS
# ==================================
def validate_arguments():
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py <image_path_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print("Error: path does not exist")
        sys.exit(1)

    return input_path


def is_image_file(path):
    return path.endswith(IMAGE_EXTENSIONS)


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        print(f"Error: unable to read image -> {path}")
        return None

    return image


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_image_info(image_path):
    file_name = os.path.basename(image_path)
    name, extension = os.path.splitext(file_name)

    class_name = os.path.basename(os.path.dirname(image_path))
    plant_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

    return plant_name, class_name, name, extension


def create_output_directory_from_image(image_path):
    plant_name, class_name, _, _ = extract_image_info(image_path)
    output_dir = os.path.join(OUTPUT_ROOT, plant_name, class_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==================================
# CORE MASK / SEGMENTATION
# ==================================
def create_binary_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    _, binary_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask


def get_largest_contour(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


# ==================================
# TRANSFORMATIONS
# ==================================
def show_gaussian_blur(image):
    return create_binary_mask(image)


def create_masked_image(image):
    binary_mask = create_binary_mask(image)

    masked = cv2.bitwise_and(image, image, mask=binary_mask)

    white_background = np.full_like(image, 255)
    inverted_mask = cv2.bitwise_not(binary_mask)
    background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

    final_masked = cv2.add(masked, background)
    return final_masked


def create_roi_objects_image(image):
    binary_mask = create_binary_mask(image)
    roi_objects = image.copy()

    height, width = binary_mask.shape[:2]
    margin_x = 3
    margin_y = 3

    x1, y1 = margin_x, margin_y
    x2, y2 = width - margin_x, height - margin_y

    roi_mask = np.zeros_like(binary_mask)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    objects_in_roi = cv2.bitwise_and(binary_mask, roi_mask)

    contours, _ = cv2.findContours(
        objects_in_roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = roi_objects.copy()
    cv2.drawContours(overlay, contours, -1, GREEN_COLOR, thickness=cv2.FILLED)

    alpha = 0.95
    roi_objects = cv2.addWeighted(overlay, alpha, roi_objects, 1 - alpha, 0)

    cv2.drawContours(roi_objects, contours, -1, GREEN_COLOR, 2)
    cv2.rectangle(roi_objects, (x1, y1), (x2, y2), BLUE_COLOR, 3)

    return roi_objects


def detect_internal_edges(image, binary_mask):
    leaf_only = cv2.bitwise_and(image, image, mask=binary_mask)
    gray_leaf = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)
    blurred_leaf = cv2.GaussianBlur(gray_leaf, (5, 5), 0)

    internal_edges = cv2.Canny(blurred_leaf, 30, 90)
    internal_edges = cv2.bitwise_and(internal_edges, binary_mask)

    kernel = np.ones((3, 3), np.uint8)
    internal_edges = cv2.dilate(internal_edges, kernel, iterations=1)

    return internal_edges


def draw_vertical_line_inside_leaf(binary_mask, cx):
    h, w = binary_mask.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(mask, (cx, 0), (cx, h - 1), 255, 3)
    mask = cv2.bitwise_and(mask, binary_mask)

    return mask


def draw_diagonal_line_inside_leaf(binary_mask, start_point, end_point):
    h, w = binary_mask.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(mask, tuple(start_point), tuple(end_point), 255, 3)
    mask = cv2.bitwise_and(mask, binary_mask)

    return mask


def create_analyze_object_image(image):
    binary_mask = create_binary_mask(image)
    analyze_image = image.copy()

    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        return analyze_image

    # contour externe en rose
    cv2.drawContours(analyze_image, [largest_contour], -1, PINK_COLOR, 3)

    # détails internes en bleu
    internal_edges = detect_internal_edges(image, binary_mask)
    analyze_image[internal_edges > 0] = BLUE_COLOR

    # centre
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return analyze_image

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    cv2.circle(analyze_image, (cx, cy), 8, PINK_COLOR, -1)
    cv2.circle(analyze_image, (cx, cy), 14, PINK_COLOR, 3)

    # ligne verticale limitée à la feuille
    vertical_mask = draw_vertical_line_inside_leaf(binary_mask, cx)
    analyze_image[vertical_mask > 0] = PINK_COLOR

    # diagonale limitée à la feuille
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    p1, p2, p3, p4 = box
    diagonals = [
        (p1, p3, np.linalg.norm(p1 - p3)),
        (p2, p4, np.linalg.norm(p2 - p4))
    ]

    diagonal_start, diagonal_end, _ = max(diagonals, key=lambda x: x[2])

    diagonal_mask = draw_diagonal_line_inside_leaf(
        binary_mask,
        diagonal_start,
        diagonal_end
    )
    analyze_image[diagonal_mask > 0] = PINK_COLOR

    return analyze_image


def create_pseudolandmarks_image(image, contour_points_count=50, internal_points_count=40):
    binary_mask = create_binary_mask(image)
    pseudo_image = image.copy()

    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        return pseudo_image

    # contour externe
    cv2.drawContours(pseudo_image, [largest_contour], -1, PINK_COLOR, 2)

    # points sur contour externe
    contour_coords = largest_contour[:, 0, :]
    total_contour_points = len(contour_coords)

    contour_step = max(1, total_contour_points // contour_points_count)
    selected_contour_points = contour_coords[::contour_step][:contour_points_count]

    for i, point in enumerate(selected_contour_points):
        x, y = point
        color = BLUE_COLOR if i < len(selected_contour_points) // 2 else PINK_COLOR
        cv2.circle(pseudo_image, (x, y), 5, color, -1)

    # points sur détails internes
    internal_edges = detect_internal_edges(image, binary_mask)
    internal_y, internal_x = np.where(internal_edges > 0)

    if len(internal_x) > 0:
        internal_coords = np.column_stack((internal_x, internal_y))
        internal_step = max(1, len(internal_coords) // internal_points_count)
        selected_internal_points = internal_coords[::internal_step][:internal_points_count]

        for x, y in selected_internal_points:
            cv2.circle(pseudo_image, (int(x), int(y)), 5, ORANGE_COLOR, -1)

    return pseudo_image


# ==================================
# HISTOGRAM
# ==================================
def create_color_histogram_figure(image):
    binary_mask = create_binary_mask(image)
    leaf = cv2.bitwise_and(image, image, mask=binary_mask)

    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(leaf, cv2.COLOR_BGR2LAB)

    fig, ax = plt.subplots(figsize=(10, 6))

    def compute_hist(img, channel, label, color):
        hist = cv2.calcHist([img], [channel], binary_mask, [256], [0, 256])
        if hist.sum() > 0:
            hist = hist / hist.sum() * 100
        ax.plot(hist, color=color, label=label)

    # RGB
    compute_hist(leaf, 0, "blue", "blue")
    compute_hist(leaf, 1, "green", "green")
    compute_hist(leaf, 2, "red", "red")

    # HSV
    compute_hist(hsv, 0, "hue", "purple")
    compute_hist(hsv, 1, "saturation", "cyan")
    compute_hist(hsv, 2, "value", "orange")

    # LAB
    compute_hist(lab, 0, "lightness", "black")
    compute_hist(lab, 1, "green-magenta", "magenta")
    compute_hist(lab, 2, "blue-yellow", "yellow")

    ax.set_title("Figure IV.7: Color Histogram")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xlim([0, 256])
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ==================================
# DISPLAY / SAVE
# ==================================
def build_transformation_panels(image):
    original_rgb = to_rgb(image)

    gaussian_blur_result = show_gaussian_blur(image)

    masked_result_rgb = to_rgb(create_masked_image(image))
    roi_objects_result_rgb = to_rgb(create_roi_objects_image(image))
    analyze_object_result_rgb = to_rgb(create_analyze_object_image(image))
    pseudolandmarks_result_rgb = to_rgb(create_pseudolandmarks_image(image))

    images = [
        original_rgb,
        gaussian_blur_result,
        masked_result_rgb,
        roi_objects_result_rgb,
        analyze_object_result_rgb,
        pseudolandmarks_result_rgb
    ]

    titles = [
        "Figure IV.1: Original",
        "Figure IV.2: Gaussian Blur",
        "Figure IV.3: Mask",
        "Figure IV.4: ROI Objects",
        "Figure IV.5: Analyze Object",
        "Figure IV.6: Pseudolandmarks"
    ]

    cmap_list = [None, "gray", None, None, None, None]

    return images, titles, cmap_list


def create_transformation_figure(image):
    images, titles, cmap_list = build_transformation_panels(image)

    fig = plt.figure(figsize=FIGURE_SIZE)

    for i in range(len(images)):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.imshow(images[i], cmap=cmap_list[i])
        ax.set_title(titles[i])
        ax.axis("on")

    fig.tight_layout()
    return fig


def display_single_image_mode(image):
    transformation_fig = create_transformation_figure(image)
    histogram_fig = create_color_histogram_figure(image)

    plt.show()


def save_figure(fig, output_path):
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_single_image_for_directory(image_path):
    image = load_image(image_path)
    if image is None:
        return

    output_dir = create_output_directory_from_image(image_path)
    _, _, name, _ = extract_image_info(image_path)

    transformation_fig = create_transformation_figure(image)
    histogram_fig = create_color_histogram_figure(image)

    transformation_output_path = os.path.join(output_dir, f"{name}_transformations.png")
    histogram_output_path = os.path.join(output_dir, f"{name}_histogram.png")

    save_figure(transformation_fig, transformation_output_path)
    save_figure(histogram_fig, histogram_output_path)

    print(f"Processed: {image_path}")
    print(f"Saved: {transformation_output_path}")
    print(f"Saved: {histogram_output_path}")
    print("-" * 60)


def process_directory(directory_path):
    found_images = 0

    for file_name in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, file_name)

        if os.path.isfile(file_path) and is_image_file(file_name):
            found_images += 1
            process_single_image_for_directory(file_path)

    if found_images == 0:
        print("No image files found in the directory.")


# ==================================
# MAIN
# ==================================
def main():
    input_path = validate_arguments()

    if os.path.isfile(input_path):
        if not is_image_file(input_path):
            print("Error: file is not a supported image.")
            sys.exit(1)

        image = load_image(input_path)
        if image is None:
            sys.exit(1)

        display_single_image_mode(image)

    elif os.path.isdir(input_path):
        process_directory(input_path)

    else:
        print("Error: invalid path")
        sys.exit(1)


if __name__ == "__main__":
    main()