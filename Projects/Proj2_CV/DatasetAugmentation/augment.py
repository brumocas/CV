import os
from PIL import Image
from torchvision.transforms import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
config = {
    "image_folder": "/home/bruno/Workspace/Master/CV/Projects/Proj2_CV/Buoy-Detection-1/train/images",
    "label_folder": "/home/bruno/Workspace/Master/CV/Projects/Proj2_CV/Buoy-Detection-1/train/labels",
    "output_image_folder": "/home/bruno/Workspace/Master/CV/Projects/Proj2_CV/DatasetAugmentation/outf",
    "output_label_folder": "/home/bruno/Workspace/Master/CV/Projects/Proj2_CV//DatasetAugmentation/outf",
    "translation_range": 100,
    "visualize": True,  # Set to True for debugging with visualization
}

# Create output directories
os.makedirs(config["output_image_folder"], exist_ok=True)
os.makedirs(config["output_label_folder"], exist_ok=True)

def random_translate(image, x_center, y_center, width, height):
    """Apply random translation to image and its bounding box."""
    dx = random.randint(-config["translation_range"], config["translation_range"])
    dy = random.randint(-config["translation_range"], config["translation_range"])

    # Apply affine transformation to the image
    transformed_image = F.affine(
        image,
        angle=0,  # No rotation
        translate=(dx, dy),
        scale=1.0, # No resize
        shear=0
    )

    # Update label for translation
    img_width, img_height = image.size

    # Update the center by translation (clamping to ensure the new coordinates stay within image bounds)
    x_center_new = max(0, min(1, x_center + dx / img_width))
    y_center_new = max(0, min(1, y_center + dy / img_height))

    return transformed_image, x_center_new, y_center_new, width, height

def visualize(image, x_center, y_center, width, height):
    """Visualize the image with the updated bounding box and center."""
    plt.imshow(image)
    img_width, img_height = image.size

    # Draw the bounding box (scaled to pixel values)
    bbox_x_min = (x_center - width / 2) * img_width
    bbox_y_min = (y_center - height / 2) * img_height
    bbox_x_max = (x_center + width / 2) * img_width
    bbox_y_max = (y_center + height / 2) * img_height

    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle(
        (bbox_x_min, bbox_y_min), bbox_x_max - bbox_x_min, bbox_y_max - bbox_y_min,
        linewidth=2, edgecolor='blue', facecolor='none'
    )
    plt.gca().add_patch(rect)

    # Draw the center (as a red dot)
    plt.scatter([x_center * img_width], [y_center * img_height], c='blue', s=50, marker='x')

    plt.title("Augmented Image with Bounding Box and Center")
    plt.show()




if __name__=="__main__":

    # Process Files
    for image_file in os.listdir(config["image_folder"]):
        image_path = os.path.join(config["image_folder"], image_file)
        label_path = os.path.join(config["label_folder"], os.path.splitext(image_file)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"Label not found for {image_file}, skipping.")
            continue

        try:
            # Load image and labels
            image = Image.open(image_path).convert("RGB")
            with open(label_path, "r") as f:
                labels = f.readlines()

            updated_labels = []
            for label in labels:
                # Parse YOLO label format
                parts = label.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Apply translation
                transformed_image, x_new, y_new, new_width, new_height = random_translate(
                    image, x_center, y_center, width, height
                )
                updated_labels.append(f"{class_id} {x_new} {y_new} {new_width} {new_height}")

            # Save the augmented image
            output_image_path = os.path.join(config["output_image_folder"], "aug_" + image_file)
            transformed_image.save(output_image_path)

            # Save the updated labels
            output_label_path = os.path.join(
                config["output_label_folder"], "aug_" + os.path.splitext(image_file)[0] + ".txt"
            )
            with open(output_label_path, "w") as f:
                f.write("\n".join(updated_labels))

            print(f"Saved augmented image and label for {image_file}.")

            # Visualize the image with bounding box and center
            if config["visualize"]:
                visualize(transformed_image, x_new, y_new, new_width, new_height)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")