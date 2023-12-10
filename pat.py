import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Function to load an image and convert it to a tensor
def image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image.size

# Function to detect peaks in each feature map
def detect_peaks(feature_map):
    peaks = []
    for i in range(feature_map.shape[1]):  # iterate over each channel
        data = feature_map[0, i, :, :].cpu().detach().numpy()
        peaks_row, _ = find_peaks(data.max(axis=1))
        peaks_col, _ = find_peaks(data.max(axis=0))
        peaks += [(r, c) for r in peaks_row for c in peaks_col]
    return peaks

# Function to calculate voting space
def compute_voting_space(peaks, grid_size):
    V = np.zeros(grid_size, dtype=np.float32)
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            if i != j:
                displacement = tuple(np.array(peaks[j]) - np.array(peaks[i]))
                V[displacement] += 1
    return V

# Function to find the best repeating step in the voting space
def find_best_repeating_step(V):
    dstar_y, dstar_x = np.unravel_index(np.argmax(V), V.shape)
    dstar = (dstar_y - V.shape[0] // 2, dstar_x - V.shape[1] // 2)
    return dstar

# Function to calculate the consistency metric for each filter
def calculate_consistency_metrics(peaks, dstar, threshold=10):
    fi_acc = np.zeros(len(peaks))
    for i, peak in enumerate(peaks):
        for other_peak in peaks:
            displacement = np.array(other_peak) - np.array(peak)
            distance = np.linalg.norm(displacement - np.array(dstar))
            if distance < threshold:
                fi_acc[i] += 1
    return fi_acc

# Function to assign weights and select filters
def assign_weights_and_select_filters(fi_acc, weight_threshold=0.5):
    weights = fi_acc / np.max(fi_acc)
    selected_filters = np.where(weights > weight_threshold)[0]
    return weights, selected_filters

# Function to accumulate displacements of consistent peaks
def accumulate_displacements(peaks, dstar, selected_filters, threshold=10):
    accum_displacements = []
    for filter_idx in selected_filters:
        for other_peak in peaks:
            displacement = np.array(other_peak) - np.array(peaks[filter_idx])
            distance = np.linalg.norm(displacement - np.array(dstar))
            if distance < threshold:
                accum_displacements.append(displacement)
    return np.mean(accum_displacements, axis=0) if accum_displacements else (0, 0)

# Function to generate bounding boxes based on the origin and step size
def generate_bounding_boxes(origin, step, num_boxes=5, box_size=(50, 50)):
    boxes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            top_left = (origin[0] + step[0] * i, origin[1] + step[1] * j)
            bottom_right = (top_left[0] + box_size[0], top_left[1] + box_size[1])
            boxes.append((top_left, bottom_right))
    return boxes

# # Function to draw bounding boxes on the image
# def draw_boxes_on_image(image_path, boxes):
#     with Image.open(image_path) as img:
#         draw = ImageDraw.Draw(img)
#         for box in boxes:
#             draw.rectangle(box, outline="red", width=2)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()

def draw_boxes_on_image(image_path, boxes, output_path):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline="red", width=2)
        img.save(output_path)

# Main function to process the image and perform all steps
def extract_features_draw_boxes(image_path):
    output_path = '/workspace/ashwinv/Repeated-Pattern-Detection-using-CNN-activations/images/out.png'
    image_tensor, original_size = image_to_tensor(image_path)
    model = alexnet(pretrained=True)
    alexnet_conv_layers = torch.nn.Sequential(*list(model.features.children()))

    features = alexnet_conv_layers(image_tensor)
    peaks = detect_peaks(features)

    grid_size = (2 * original_size[0], 2 * original_size[1])
    V = compute_voting_space(peaks, grid_size)
    V_smoothed = gaussian_filter(V, sigma=5)
    dstar = find_best_repeating_step(V_smoothed)
    fi_acc = calculate_consistency_metrics(peaks, dstar)
    weights, selected_filters = assign_weights_and_select_filters(fi_acc)
    origin_estimate = accumulate_displacements(peaks, dstar, selected_filters)

    boxes = generate_bounding_boxes(origin_estimate, dstar, num_boxes=5, box_size=(50, 50))

    draw_boxes_on_image(image_path, boxes, output_path)
    return features, V_smoothed, dstar, fi_acc, weights, selected_filters, origin_estimate, boxes

# Example usage
image_path = '/workspace/ashwinv/Repeated-Pattern-Detection-using-CNN-activations/p4.png'  # Replace with your image path

results = extract_features_draw_boxes(image_path)

