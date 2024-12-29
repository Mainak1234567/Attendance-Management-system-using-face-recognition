import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Attendance System")
        self.root.geometry("800x600")
        self.setup_gui()

    def setup_gui(self):
        ttk.Label(self.root, text="Face Recognition Attendance System", font=("Arial", 16)).pack(pady=20)

        ttk.Button(self.root, text="Train Images", command=self.train_images).pack(pady=10)
        ttk.Button(self.root, text="Take Attendance", command=self.start_attendance_thread).pack(pady=10)
        ttk.Button(self.root, text="Quit", command=self.root.quit).pack(pady=10)

    def ensure_directories(self):
        os.makedirs("TrainingImages", exist_ok=True)
        os.makedirs("Attendance", exist_ok=True)
        os.makedirs("TrainedModel", exist_ok=True)

    def train_images(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            def get_images_and_labels(path):
                image_paths = [os.path.join(path, f) for f in os.listdir(path)]
                face_samples = []
                ids = []
                for image_path in image_paths:
                    try:
                        pil_image = Image.open(image_path).convert('L')
                        image_np = np.array(pil_image, 'uint8')
                        id_ = int(os.path.split(image_path)[-1].split(".")[1])
                        faces = detector.detectMultiScale(image_np)
                        for (x, y, w, h) in faces:
                            face_samples.append(image_np[y:y + h, x:x + w])
                            ids.append(id_)
                    except Exception as e:
                        logging.error(f"Error processing {image_path}: {e}")
                return face_samples, ids

            self.ensure_directories()
            faces, ids = get_images_and_labels("TrainingImages")
            recognizer.train(faces, np.array(ids))
            recognizer.save("TrainedModel/trainer.yml")
            messagebox.showinfo("Success", "Images trained successfully.")

        except Exception as e:
            logging.error(f"Error in training images: {e}")
            messagebox.showerror("Error", "Failed to train images.")

    def start_attendance_thread(self):
        thread = threading.Thread(target=self.take_attendance)
        thread.start()

    def take_attendance(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("TrainedModel/trainer.yml")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            self.ensure_directories()

            attendance_file = os.path.join("Attendance", "attendance.csv")
            if not os.path.exists(attendance_file):
                pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(attendance_file, index=False)

            id_name_mapping = {1: "Alice", 2: "Bob"}  # Replace with a proper mapping or load from a file
            df = pd.read_csv(attendance_file)

            cam = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            try:
                while True:
                    ret, img = cam.read()
                    if not ret:
                        logging.warning("Failed to access the camera.")
                        break

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

                    for (x, y, w, h) in faces:
                        id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        if confidence < 100:
                            name = id_name_mapping.get(id_, f"User_{id_}")
                            timestamp = pd.Timestamp.now()
                            new_entry = {"ID": id_, "Name": name, "Date": timestamp.date(), "Time": timestamp.time()}
                            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                            df.to_csv(attendance_file, index=False)
                            cv2.putText(img, name, (x, y - 10), font, 0.75, (255, 255, 255), 2)
                        else:
                            cv2.putText(img, "Unknown", (x, y - 10), font, 0.75, (255, 0, 0), 2)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

                    cv2.imshow('Attendance', img)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            finally:
                cam.release()
                cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error in taking attendance: {e}")
            messagebox.showerror("Error", "Failed to take attendance.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
