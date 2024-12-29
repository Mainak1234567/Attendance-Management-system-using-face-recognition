import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("800x600")
        self.setup_gui()

    def setup_gui(self):
        """
        Sets up the graphical user interface.
        """
        ttk.Label(self.root, text="Face Recognition Attendance System", font=("Arial", 16)).pack(pady=20)

        ttk.Button(self.root, text="Train Images", command=self.train_images).pack(pady=10)
        ttk.Button(self.root, text="Take Attendance", command=self.take_attendance).pack(pady=10)
        ttk.Button(self.root, text="Quit", command=self.root.quit).pack(pady=10)

    def train_images(self):
        """
        Calls the train.py script to train images.
        """
        try:
            os.system("python train.py")  # Execute the train.py script
            messagebox.showinfo("Success", "Images trained successfully.")
        except Exception as e:
            logging.error(f"Error in training images: {e}")
            messagebox.showerror("Error", "Failed to train images.")

    def take_attendance(self):
        """
        Takes attendance using the trained face recognition model.
        """
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("TrainedModel/trainer.yml")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            attendance_file = os.path.join("Attendance", "attendance.csv")
            os.makedirs("Attendance", exist_ok=True)

            if not os.path.exists(attendance_file):
                pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(attendance_file, index=False)

            attendance_data = pd.read_csv(attendance_file)

            cam = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    if confidence < 100:
                        name = f"User_{id_}"  # Replace with actual name lookup logic
                        timestamp = datetime.now()
                        date_str = timestamp.strftime("%Y-%m-%d")
                        time_str = timestamp.strftime("%H:%M:%S")

                        if not attendance_data[(attendance_data["ID"] == id_) & (attendance_data["Date"] == date_str)].empty:
                            continue  # Skip if already recorded for the day

                        new_entry = {"ID": id_, "Name": name, "Date": date_str, "Time": time_str}
                        attendance_data = attendance_data.append(new_entry, ignore_index=True)
                        attendance_data.to_csv(attendance_file, index=False)

                        cv2.putText(img, f"{name}", (x, y - 10), font, 0.75, (255, 255, 255), 2)
                    else:
                        cv2.putText(img, "Unknown", (x, y - 10), font, 0.75, (0, 0, 255), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

                cv2.imshow("Attendance", img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            cam.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Info", "Attendance has been recorded.")
        except Exception as e:
            logging.error(f"Error in taking attendance: {e}")
            messagebox.showerror("Error", "Failed to take attendance.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
