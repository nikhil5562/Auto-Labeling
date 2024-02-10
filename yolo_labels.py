import os
import shutil
import torch
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-face.pt')

# Base directory for the dataset
base_dataset_directory = r'C:\Users\nikhi\Desktop\yolov8-face'

# Create 'dataset', 'images', and 'labels' directories if they do not exist
images_dir = os.path.join(base_dataset_directory, 'dataset', 'images')
labels_dir = os.path.join(base_dataset_directory, 'dataset', 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Assuming each subdirectory in 'dataset_directory' is named after an emotion
dataset_directory = os.path.join(base_dataset_directory, 'affectnet_dataset_balanced_batch3')

# Map emotions to class indices
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotions)}

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = x_min + width / 2, y_min + height / 2
    return x_center/img_width, y_center/img_height, width/img_width, height/img_height

# Process each subdirectory
for emotion in emotions:
    subdirectory_path = os.path.join(dataset_directory, emotion)
    for image_name in os.listdir(subdirectory_path):
        image_path = os.path.join(subdirectory_path, image_name)
        
        # Inference
        results = model(image_path)

        # Process each detection in the results
        for result in results:
            # Access the tensor with bounding box data
            if result.boxes is not None:
                detections = result.boxes.boxes
                img_width, img_height = result.orig_shape

                # Copy the image to the 'images' subdirectory
                shutil.copy(image_path, os.path.join(images_dir, image_name))

                # Iterate through each bounding box
                for detection in detections:
                    bbox = detection[:4].numpy()  # Convert to numpy array
                    normalized_bbox = convert_to_yolo_format(bbox, img_width, img_height)
                    label_line = f"{emotion_to_index[emotion]} {' '.join(map(str, normalized_bbox))}\n"

                    # Save the label in a .txt file in the 'labels' subdirectory
                    label_name = os.path.splitext(image_name)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_name)
                    with open(label_path, 'w') as label_file:
                        label_file.write(label_line)
        
        # Optional: Print progress
        print(f"Processed {image_name}")

print("Conversion to YOLO format completed!")
