# gesture_app_tk.py
# Advanced Gesture Controlled Application (Modern Tkinter UI)
# Single-file full version

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import json
import threading
import time
import os
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ================= CONFIG =================
GESTURES = [
    "Open Palm", "Fist", "Peace", "Thumbs Up", "Thumbs Down",
    "Pointing", "Swipe Left", "Swipe Right", "Unknown", "No Hand"
]

DEFAULT_MAPPING = {
    "Open Palm": "None",
    "Fist": "Stop",
    "Peace": "Play/Pause",
    "Thumbs Up": "Volume Up",
    "Thumbs Down": "Volume Down",
    "Pointing": "Next",
    "Swipe Left": "Previous",
    "Swipe Right": "Next",
    "Unknown": "None",
    "No Hand": "None"
}

MAPPING_FILE = "gestures.json"

DEBOUNCE_FRAMES = 5
SWIPE_WINDOW = 8
SWIPE_MIN_DISTANCE = 0.18
TRIGGER_COOLDOWN = 1.0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= HELPERS =================
def load_mapping():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_MAPPING.copy()

def save_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

def perform_action(action):
    if action == "None":
        return
    try:
        if action == "Volume Up":
            pyautogui.press("volumeup")
        elif action == "Volume Down":
            pyautogui.press("volumedown")
        elif action == "Play/Pause":
            pyautogui.press("playpause")
        elif action == "Next":
            pyautogui.press("nexttrack")
        elif action == "Previous":
            pyautogui.press("prevtrack")
        elif action == "Stop":
            pyautogui.press("stop")
    except:
        pass

def landmarks_to_np(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

def normalize_landmarks(coords):
    coords = coords.copy()
    center = coords[0, :2]
    coords[:, :2] -= center
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    coords[:, :2] /= scale if scale != 0 else 1
    return coords

def finger_states(coords):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = [False]*5
    for i in range(1,5):
        states[i] = coords[tips[i],1] < coords[pips[i],1]
    states[0] = coords[4,0] < coords[3,0]
    return states

def classify_gesture(coords):
    states = finger_states(coords)
    if all(states):
        return "Open Palm"
    if not any(states):
        return "Fist"
    if states[1] and states[2] and not any([states[0], states[3], states[4]]):
        return "Peace"
    if states[0] and not any(states[1:]):
        return "Thumbs Up" if coords[4,1] < coords[0,1] else "Thumbs Down"
    if states[1] and not any([states[0], states[2], states[3], states[4]]):
        return "Pointing"
    return "Unknown"

# ================= MAIN APP =================
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Control System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.mapping = load_mapping()
        self.running = False
        self.detected_queue = deque(maxlen=DEBOUNCE_FRAMES)
        self.centroid_history = deque(maxlen=SWIPE_WINDOW)
        self.last_trigger = defaultdict(lambda: 0)

        self.setup_ui()

        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # ================= UI =================
    def setup_ui(self):
        sidebar = tk.Frame(self.root, bg="#111111", width=220)
        sidebar.pack(side="left", fill="y")

        tk.Label(sidebar, text="Gesture App",
                 fg="white", bg="#111111",
                 font=("Segoe UI", 18, "bold")).pack(pady=20)

        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(side="right", fill="both", expand=True)

        self.video_label = tk.Label(main, bg="black")
        self.video_label.pack(padx=10, pady=10)

        info = tk.Frame(main, bg="#2a2a2a")
        info.pack(fill="x", padx=20)

        self.status_var = tk.StringVar(value="Stopped")
        self.gesture_var = tk.StringVar(value="No Hand")
        self.fps_var = tk.StringVar(value="0")

        self.info_block(info, "Status", self.status_var)
        self.info_block(info, "Gesture", self.gesture_var)
        self.info_block(info, "FPS", self.fps_var)

        btns = tk.Frame(main, bg="#1e1e1e")
        btns.pack(pady=20)

        tk.Button(btns, text="▶ Start", bg="#28a745",
                  fg="white", width=12,
                  command=self.start).pack(side="left", padx=10)

        tk.Button(btns, text="⏹ Stop", bg="#dc3545",
                  fg="white", width=12,
                  command=self.stop).pack(side="left", padx=10)

    def info_block(self, parent, title, var):
        frame = tk.Frame(parent, bg="#2a2a2a")
        frame.pack(side="left", padx=30, pady=10)
        tk.Label(frame, text=title, fg="#aaaaaa",
                 bg="#2a2a2a").pack()
        tk.Label(frame, textvariable=var,
                 fg="white", bg="#2a2a2a",
                 font=("Segoe UI", 14, "bold")).pack()

    # ================= LOGIC =================
    def start(self):
        if self.running:
            return
        self.running = True
        self.status_var.set("Running")
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.status_var.set("Stopped")

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        prev = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            gesture = "No Hand"
            centroid_x = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                coords = normalize_landmarks(landmarks_to_np(hand))
                gesture = classify_gesture(coords)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                centroid_x = coords[0,0]

            if centroid_x is not None:
                self.centroid_history.append(centroid_x)

            if len(self.centroid_history) >= SWIPE_WINDOW:
                dx = self.centroid_history[-1] - self.centroid_history[0]
                if dx > SWIPE_MIN_DISTANCE:
                    gesture = "Swipe Right"
                elif dx < -SWIPE_MIN_DISTANCE:
                    gesture = "Swipe Left"

            self.detected_queue.append(gesture)

            if len(self.detected_queue) == DEBOUNCE_FRAMES and len(set(self.detected_queue)) == 1:
                now = time.time()
                if now - self.last_trigger[gesture] > TRIGGER_COOLDOWN:
                    self.last_trigger[gesture] = now
                    perform_action(self.mapping.get(gesture, "None"))

            fps = int(1 / (time.time() - prev))
            prev = time.time()

            self.root.after(0, self.update_ui, frame, fps, gesture)

        cap.release()

    def update_ui(self, frame, fps, gesture):
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.configure(image=img)
        self.video_label.image = img
        self.gesture_var.set(gesture)
        self.fps_var.set(str(fps))

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
