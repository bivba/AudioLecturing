import cv2
import os
import numpy as np
import mss
import time
from skimage.metrics import structural_similarity as ssim
from threading import Thread, Event

class VideoRecorder:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        #self.sct = mss.mss()
        self.monitor = mss.mss().monitors[2]
        self.last = None
        self.index = 0
        self.out = out_dir
        self.recording = False
        self.thread = None
        self.stop_event = Event()

    def record(self):
        self.recording = True
        self.thread = Thread(target=self._record_loop)
        self.thread.start()

    def _record_loop(self):
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                screenshot = sct.grab(self.monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.last is not None:
                    score = ssim(self.last, gray)
                    if score < 0.95:  # SSIM threshold for slide change detection
                        filename = os.path.join(self.out, f"slide_{self.index}.png")
                        self.index += 1
                        cv2.imwrite(filename, frame)

                self.last = gray
                time.sleep(5)  # ~3 fps

    def stop(self):
        self.recording = False
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def get_screenshots(self):
        screenshots = []
        for i in range(self.index):
            path = os.path.join(self.out, f"slide_{i}.png")
            if os.path.exists(path):
                screenshots.append(path)
        return screenshots

    def release(self):
        self.stop()