import cv2
import threading
from queue import Queue

class CameraThread(threading.Thread):
    def __init__(self, camera_path, frame_queue):
        threading.Thread.__init__(self)
        self.camera_path = camera_path
        self.frame_queue = frame_queue
        self.cap = cv2.VideoCapture(camera_path)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame.")
                continue
            self.frame_queue.put(frame)

    def stop(self):
        self.running = False
        self.cap.release()