import os
import glob as glob
import matplotlib.pyplot as plt
import random
import cv2
import re
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_label_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Regular expression to capture numbers including spaces and new lines
    pattern = re.compile(r'(\d+)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)')
    matches = pattern.findall(data)
    
    # Convert the matched groups to a list of tuples with floats
    bounding_boxes = [(int(cls), float(x1), float(y1), float(x2), float(y2)) for cls, x1, y1, x2, y2 in matches]
    
    return bounding_boxes

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels,class_names):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)

        thickness = max(2, int(w/275))
                
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )

        # Draw the class label text
        label_text = class_names[labels[box_num]]
        font_scale = max(0.5, w / 1000)
        cv2.putText(
            image, 
            label_text, 
            (xmin, ymin - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 0), 
            thickness=thickness // 2
        )
    return image

# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples,class_names):
    image_files = sorted(glob.glob(os.path.join(image_paths, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_paths, '*.txt')))
    
    # Create dictionaries to map base filenames to full paths
    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    label_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

    # Find common keys (base filenames)
    common_keys = list(set(image_dict.keys()) & set(label_dict.keys()))
    common_keys.sort()  # Sort to maintain consistency
    matched_images = [image_dict[key] for key in common_keys]
    matched_labels = [label_dict[key] for key in common_keys]
    
    num_images = len(matched_images)
    print(f'Found {num_images} matching image-label pairs.')

    plt.figure(figsize=(9, 6))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(matched_images[j])
        bboxes = []
        labels = []
        result = parse_label_file(matched_labels[j])
        for k in result:
            bboxes.append([k[1], k[2], k[3], k[4]])
            labels.append(k[0])
        result_image = plot_box(image, bboxes, labels,class_names)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()
    return matched_images,matched_labels