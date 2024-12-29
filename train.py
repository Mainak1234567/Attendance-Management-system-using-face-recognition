import cv2
import numpy as np
import os
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ensure_directories():
    """
    Ensures necessary directories exist for training images and the trained model.
    """
    os.makedirs("TrainingImages", exist_ok=True)
    os.makedirs("TrainedModel", exist_ok=True)

def get_images_and_labels(path):
    """
    Reads images and their associated labels from a specified directory.

    Args:
        path (str): Path to the directory containing training images.

    Returns:
        tuple: A tuple containing face samples and their corresponding IDs.
    """
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    face_samples = []
    ids = []

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert('L')  # Convert image to grayscale
            image_np = np.array(pil_image, 'uint8')
            id_ = int(os.path.split(image_path)[-1].split(".")[1])  # Extract ID from file name
            faces = detector.detectMultiScale(image_np)

            for (x, y, w, h) in faces:
                face_samples.append(image_np[y:y + h, x:x + w])
                ids.append(id_)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
    
    return face_samples, ids

def train_model():
    """
    Trains the LBPH face recognizer with the images and labels from the TrainingImages directory
    and saves the trained model.
    """
    try:
        ensure_directories()
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = get_images_and_labels("TrainingImages")

        if not faces:
            logging.error("No faces found in the TrainingImages directory. Ensure images are in place.")
            return

        recognizer.train(faces, np.array(ids))
        model_path = "TrainedModel/trainer.yml"
        recognizer.save(model_path)

        logging.info(f"Model trained successfully and saved at {model_path}.")
        print("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")

if __name__ == "__main__":
    train_model()
