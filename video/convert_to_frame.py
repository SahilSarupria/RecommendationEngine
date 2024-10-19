import cv2
import torch
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained YOLOv5 model (this example uses YOLOv5 from Ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set a confidence threshold for YOLOv5 detections
CONFIDENCE_THRESHOLD = 0.5

def detect_faces_and_objects(frame):
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)

    # Use the YOLOv5 model to detect objects
    results = model(frame)
    # Filter out results based on confidence
    objects = results.xyxy[0]  # Predictions (x1, y1, x2, y2, confidence, class)
    object_count = sum(1 for obj in objects if obj[4] >= CONFIDENCE_THRESHOLD)

    return face_count, object_count

def extract_best_frame(video_path, output_path, skip_frames=5, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    
    best_frame = None
    max_faces = 0
    max_objects = 0
    max_combined = 0  # Initialize max combined count to an impossible low value
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_to_process = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on the specified factor
        if i % skip_frames != 0:
            continue
        
        # Resize frame for faster processing
        if resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

        frames_to_process.append(frame)

    cap.release()

    # Process frames in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(detect_faces_and_objects, frames_to_process))

    # Determine the best frame based on face and object counts
    for i, (face_count, object_count) in enumerate(results):
        # Prioritize faces: weight face count more than object count
        combined_count = face_count * 2 + object_count  # Giving double weight to face count
        
        if combined_count > max_combined:
            max_combined = combined_count
            max_faces = face_count
            max_objects = object_count
            best_frame = frames_to_process[i]
    
    # Save the best frame
    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        print(f"Best frame with {max_faces} faces and {max_objects} objects saved at: {output_path}")
    else:
        print("No frames extracted.")

# Example usage
video_path = 'video.mp4'
output_path = 'videobet.jpg'
extract_best_frame(video_path, output_path, skip_frames=5, resize_factor=0.5)



"""import cv2
import torch
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained YOLOv5 model (this example uses YOLOv5 from Ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set a confidence threshold for YOLOv5 detections
CONFIDENCE_THRESHOLD = 0.5

# Load a pre-trained model for feature extraction (ResNet)
feature_extractor = models.resnet50(pretrained=True)
feature_extractor.eval()  # Set the model to evaluation mode

# Define the necessary transformations for the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")  # Adjust the URL as needed
collection_name = "video_recommendations"  # Your Qdrant collection name

def detect_faces_and_objects(frame):
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)

    # Use the YOLOv5 model to detect objects
    results = model(frame)
    # Filter out results based on confidence
    objects = results.xyxy[0]  # Predictions (x1, y1, x2, y2, confidence, class)
    object_count = sum(1 for obj in objects if obj[4] >= CONFIDENCE_THRESHOLD)

    return face_count, object_count

def extract_best_frame(video_path, output_path, skip_frames=5, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    
    best_frame = None
    max_faces = 0
    max_objects = 0
    max_combined = 0  # Initialize max combined count to an impossible low value
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_to_process = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on the specified factor
        if i % skip_frames != 0:
            continue
        
        # Resize frame for faster processing
        if resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

        frames_to_process.append(frame)

    cap.release()

    # Process frames in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(detect_faces_and_objects, frames_to_process))

    # Determine the best frame based on face and object counts
    for i, (face_count, object_count) in enumerate(results):
        # Prioritize faces: weight face count more than object count
        combined_count = face_count * 2 + object_count  # Giving double weight to face count
        
        if combined_count > max_combined:
            max_combined = combined_count
            max_faces = face_count
            max_objects = object_count
            best_frame = frames_to_process[i]
    
    # Save the best frame
    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        print(f"Best frame with {max_faces} faces and {max_objects} objects saved at: {output_path}")
        return best_frame  # Return the best frame for feature extraction
    else:
        print("No frames extracted.")
        return None

def image_to_vector(image):
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():  # Disable gradient calculation
        features = feature_extractor(image_tensor)
    
    return features.flatten().numpy()  # Flatten to 1D vector

def upload_vector_to_qdrant(vector, video_id):
    point = PointStruct(id=video_id, vector=vector.tolist())  # Convert to list for Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=[point])
    print(f"Vector uploaded to Qdrant for video ID: {video_id}")

# Example usage
video_path = 'video.mp4'
output_path = 'videobet.jpg'
best_frame = extract_best_frame(video_path, output_path, skip_frames=5, resize_factor=0.5)

if best_frame is not None:
    # Convert the best frame into a feature vector
    feature_vector = image_to_vector(best_frame)
    
    # Upload the feature vector to Qdrant
    upload_vector_to_qdrant(feature_vector, video_id=1)  # Assign an ID to the video (e.g., 1)
"""
