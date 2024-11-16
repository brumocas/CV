import os

def count_unique_class_labels(folder_path):
    unique_class_labels = set()
    
    # List all .txt files in the specified folder
    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and os.path.isfile(os.path.join(folder_path, f))]
    
    # Iterate through each file
    count = 0
    for file_name in label_files:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r') as file:
            for line in file:
                # Split line into components and check if it has exactly 5 values
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        print(parts)
                        # The first part is the class label (integer)
                        class_label = int(parts[0])  # Convert the first part to integer
                        unique_class_labels.add(class_label)  # Add the class label to the set
                        count = count + 1
                    except ValueError:
                        print(f"Invalid line in file {file_name}: {line.strip()}")
    
    # Output the total count of unique class labels
    print(f"Total number of unique class labels in the dataset: {len(unique_class_labels)}")
    print(f"Total number of images in the dataset: {count}")
    return unique_class_labels

# Specify your folder path here
folder_path = 'Buoy-Detection-1/valid/labels'
unique_class_labels = count_unique_class_labels(folder_path)

# Optional: Print all unique class labels if needed
# print("Unique class labels in the dataset:")
# for label in unique_class_labels:
#     print(label)


