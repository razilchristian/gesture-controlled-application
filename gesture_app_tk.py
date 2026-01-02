# ==========================================================
# Advanced Gesture Controlled Application (Tkinter)
# Author: Razil Christian
# Semester: BTech AIML – 3rd Sem
# ==========================================================

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

# ================= MEDIAPIPE INIT =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= HELPERS =================
def load_mapping():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_MAPPING.copy()

def save_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

def perform_action(action):
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

def landmarks_to_np(hand):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])

def normalize(coords):
    center = coords[0][:2]
    coords[:, :2] -= center
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if scale > 0:
        coords[:, :2] /= scale
    return coords

def finger_states(coords):
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    states = [False]*5
    for i in range(1,5):
        states[i] = coords[tips[i]][1] < coords[pip[i]][1]
    states[0] = coords[4][0] < coords[3][0]
    return states

def classify(coords):
    states = finger_states(coords)
    if all(states): return "Open Palm"
    if not any(states): return "Fist"
    if states[1] and states[2] and not any([states[0],states[3],states[4]]):
        return "Peace"
    if states[0] and not any(states[1:]):
        return "Thumbs Up" if coords[4][1] < coords[0][1] else "Thumbs Down"
    if states[1] and not any([states[0],states[2],states[3],states[4]]):
        return "Pointing"
    return "Unknown"

# ================= GUI APP =================
class GestureApp:
    def __init__(self, root):
        self.root = root
        root.title("Gesture Controlled Application")
        root.geometry("1100x600")
        root.configure(bg="#1e1e1e")

        self.mapping = load_mapping()
        self.running = False
        self.detected_queue = deque(maxlen=DEBOUNCE_FRAMES)
        self.centroid_history = deque(maxlen=SWIPE_WINDOW)
        self.last_trigger = defaultdict(float)

        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # ========== UI ==========
        self.video_label = ttk.Label(root)
        self.video_label.place(x=20, y=20)

        ttk.Label(root, text="Detected Gesture:", font=("Segoe UI", 12)).place(x=850, y=40)
        self.gesture_var = tk.StringVar(value="No Hand")
        ttk.Label(root, textvariable=self.gesture_var, font=("Segoe UI", 14, "bold")).place(x=850, y=70)

        self.start_btn = ttk.Button(root, text="▶ Start", command=self.start)
        self.start_btn.place(x=850, y=120)

        self.stop_btn = ttk.Button(root, text="⏹ Stop", command=self.stop, state="disabled")
        self.stop_btn.place(x=940, y=120)

        ttk.Label(root, text="Gesture Mapping", font=("Segoe UI", 13, "bold")).place(x=830, y=180)

        self.combo = {}
        y = 220
        for g in GESTURES:
            ttk.Label(root, text=g).place(x=800, y=y)
            v = tk.StringVar(value=self.mapping[g])
            cb = ttk.Combobox(root, textvariable=v, width=15,
                              values=list(set(DEFAULT_MAPPING.values())))
            cb.place(x=950, y=y)
            self.combo[g] = v
            y += 30

        ttk.Button(root, text="💾 Save", command=self.save).place(x=900, y=y+10)

    def start(self):
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def save(self):
        for g in GESTURES:
            self.mapping[g] = self.combo[g].get()
        save_mapping(self.mapping)
        messagebox.showinfo("Saved", "Gesture mapping saved")

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            gesture = "No Hand"
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                coords = normalize(landmarks_to_np(hand))
                gesture = classify(coords)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            self.detected_queue.append(gesture)
            if len(set(self.detected_queue)) == 1:
                now = time.time()
                if now - self.last_trigger[gesture] > TRIGGER_COOLDOWN:
                    self.last_trigger[gesture] = now
                    perform_action(self.mapping.get(gesture, "None"))

            self.gesture_var.set(gesture)

            img = ImageTk.PhotoImage(Image.fromarray(rgb).resize((760,430)))
            self.video_label.imgtk = img
            self.video_label.config(image=img)

        cap.release()

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
