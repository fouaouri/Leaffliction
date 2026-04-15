import sys
import os
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
BALANCE_THRESHOLD = 2


def validate_arguments():
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not os.path.isdir(dataset_path):
        print("Error: path does not exist or is not a directory")
        sys.exit(1)

    return dataset_path


def get_plant_name(dataset_path):
    return os.path.basename(dataset_path)


def count_images_in_class(class_path):
    image_count = 0

    for file_name in os.listdir(class_path):
        if file_name.endswith(IMAGE_EXTENSIONS):
            image_count += 1

    return image_count


def get_class_counts(dataset_path):
    class_counts = {}

    for item in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, item)

        if os.path.isdir(class_path):
            class_counts[item] = count_images_in_class(class_path)

    return class_counts


def print_dataset_statistics(plant_name, class_counts):
    print(f"Plant type: {plant_name}")
    print("Number of images per class:")

    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count}")

    total_images = sum(class_counts.values())
    print(f"-> Total images: {total_images}")

    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)

    print(f"Smallest class: {min_class} - {class_counts[min_class]}")
    print(f"Largest class: {max_class} - {class_counts[max_class]}")

    if class_counts[max_class] > BALANCE_THRESHOLD * class_counts[min_class]:
        print("Dataset is imbalanced")
    else:
        print("Dataset is balanced")


def plot_bar_chart(plant_name, class_counts):
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = plt.cm.tab20.colors

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=colors[:len(labels)])

    plt.title(f"{plant_name} - Distribution of images by class")
    plt.xlabel("Classes")
    plt.ylabel("Number of images")

    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_pie_chart(plant_name, class_counts):
    labels = list(class_counts.keys())
    values = list(class_counts.values())

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(f"{plant_name} - Class distribution")
    plt.tight_layout()


def main():
    dataset_path = validate_arguments()
    plant_name = get_plant_name(dataset_path)
    class_counts = get_class_counts(dataset_path)

    if not class_counts:
        print("Error: no class directories found in the dataset")
        sys.exit(1)

    print_dataset_statistics(plant_name, class_counts)
    plot_bar_chart(plant_name, class_counts)
    plot_pie_chart(plant_name, class_counts)

    plt.show()
if __name__ == "__main__":
    main()