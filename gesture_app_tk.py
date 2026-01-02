# gesture_app_tk.py
# Advanced Gesture Controlled Application (Tkinter)
# Features:
# - OpenCV webcam capture
# - MediaPipe Hands (multi-hand)
# - Gesture classification: Open Palm, Fist, Peace, Thumbs Up, Thumbs Down, Pointing
# - Swipe left / right detection (centroid motion)
# - Debounce/smoothing, cooldown
# - Tkinter GUI: live video, FPS, detected gesture, mapping editor, start/stop, settings
# - Uses gestures.json for mapping (load/save)
#
# Requirements: opencv-python mediapipe numpy pyautogui Pillow

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
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ---------- Config / Defaults ----------
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
    "No Hand": "None",
    
}
MAPPING_FILE = "gestures.json"

# smoothing & thresholds
DEBOUNCE_FRAMES = 5
SWIPE_WINDOW = 8  # frames to consider for swipe
SWIPE_MIN_DISTANCE = 0.18  # normalized units
TRIGGER_COOLDOWN = 1.0  # seconds between actions

# mediapipe init
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------- Helpers ----------

def load_mapping(path=MAPPING_FILE):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                d = json.load(f)
            # ensure all gestures exist
            for g in GESTURES:
                if g not in d:
                    d[g] = DEFAULT_MAPPING.get(g, "None")
            return d
        except Exception:
            pass
    return DEFAULT_MAPPING.copy()

def save_mapping(d, path=MAPPING_FILE):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def perform_action(action_name):
    if not action_name or action_name == "None":
        return
    try:
        # common media actions (may vary by OS)
        if action_name == "Volume Up":
            pyautogui.press("volumeup")
        elif action_name == "Volume Down":
            pyautogui.press("volumedown")
        elif action_name == "Play/Pause":
            pyautogui.press("playpause")
        elif action_name == "Next":
            pyautogui.press("nexttrack")
        elif action_name == "Previous":
            pyautogui.press("prevtrack")
        elif action_name == "Stop":
            # send media stop key (not supported everywhere)
            pyautogui.press("stop")
        elif action_name == "tounge":
            pyautogui.press("tell me a joke")
        else:
            # allow hotkey combos like "ctrl+alt+p"
            if "+" in action_name:
                keys = action_name.split("+")
                pyautogui.hotkey(*keys)
            else:
                # fallback: try pressing single key
                pyautogui.press(action_name)
    except Exception:
        # ignore action errors; best-effort
        pass

def landmarks_to_np(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)  # (21,3)

def normalize_landmarks(coords):
    # coords: (21,2) or (21,3)
    coords = coords.copy()
    # center on wrist (index 0)
    center = coords[0, :2].copy()
    coords[:, :2] -= center
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    coords[:, :2] /= scale
    return coords

def finger_states(coords2):  # coords2 normalized (21,2 or 21,3)
    # Return list of booleans [thumb, index, middle, ring, pinky] True if extended
    # Use simple heuristics comparing tip vs pip in y & thumb x depending on hand laterality
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    states = [False]*5
    for i in range(1,5):
        states[i] = coords2[tips[i],1] < coords2[pip[i],1]  # smaller y = higher = extended
    # thumb heuristic uses x (thumb points sideways); we will correct handedness outside
    states[0] = coords2[tips[0],0] < coords2[pip[0],0]
    return states

def angle_between(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return 0.0
    cos = np.dot(a, b) / (na*nb)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def classify_gesture(coords3, handedness_label="Right"):
    # coords3: (21,3) normalized
    coords2 = coords3[:, :2]
    states = finger_states(coords3)  # thumb heuristic may need handedness adjust
    if handedness_label == "Left":
        # thumb x-direction flip
        states[0] = coords3[4,0] > coords3[3,0]
    else:
        states[0] = coords3[4,0] < coords3[3,0]

    # Simple rules:
    if all(states):
        return "Open Palm"
    if not any(states):
        return "Fist"
    # peace (index + middle)
    if states[1] and states[2] and not states[0] and not states[3] and not states[4]:
        return "Peace"
    # thumbs up/down: thumb only, determine up/down by comparing tip y vs wrist y
    if states[0] and not any(states[1:]):
        # thumb direction: compare tip y to wrist y (smaller is up)
        wrist_y = coords3[0,1]
        thumb_tip_y = coords3[4,1]
        if thumb_tip_y < wrist_y - 0.05:
            return "Thumbs Up"
        else:
            return "Thumbs Down"
    # pointing: only index extended
    if states[1] and not any([states[0], states[2], states[3], states[4]]):
        return "Pointing"
    return "Unknown"

# ---------- GUI App ----------

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Controller (Tkinter)")

        # video frame
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=0, rowspan=8, padx=8, pady=8)

        # status/info
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(root, text="Status:").grid(row=0, column=1, sticky="w")
        ttk.Label(root, textvariable=self.status_var).grid(row=0, column=2, sticky="w")

        self.gesture_var = tk.StringVar(value="No Hand")
        ttk.Label(root, text="Detected:").grid(row=1, column=1, sticky="w")
        ttk.Label(root, textvariable=self.gesture_var).grid(row=1, column=2, sticky="w")

        self.fps_var = tk.StringVar(value="0")
        ttk.Label(root, text="FPS:").grid(row=2, column=1, sticky="w")
        ttk.Label(root, textvariable=self.fps_var).grid(row=2, column=2, sticky="w")

        # mapping editor
        ttk.Label(root, text="Gesture Mapping:").grid(row=3, column=1, sticky="w")
        self.mapping = load_mapping()
        self.map_vars = {}
        actions = ["None", "Volume Up", "Volume Down", "Play/Pause", "Next", "Previous", "Stop"]  # can add hotkeys
        for i, g in enumerate(GESTURES):
            lbl = ttk.Label(root, text=g)
            lbl.grid(row=4+i, column=1, sticky="w")
            v = tk.StringVar(value=self.mapping.get(g, "None"))
            self.map_vars[g] = v
            cmb = ttk.Combobox(root, textvariable=v, values=actions, width=18)
            cmb.grid(row=4+i, column=2, sticky="w")

        # control buttons
        self.start_btn = ttk.Button(root, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=3, padx=6)
        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=1, column=3, padx=6)
        self.save_btn = ttk.Button(root, text="Save Mapping", command=self.save_mapping)
        self.save_btn.grid(row=2, column=3, padx=6)
        self.settings_btn = ttk.Button(root, text="Settings", command=self.open_settings)
        self.settings_btn.grid(row=3, column=3, padx=6)

        # internal state
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame_time = None
        self.mapping = load_mapping()
        self.detected_queue = deque(maxlen=DEBOUNCE_FRAMES)
        self.centroid_history = deque(maxlen=SWIPE_WINDOW)
        self.last_trigger_time = defaultdict(lambda: 0.0)
        self.mp_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        # refresh mapping from UI
        for g in GESTURES:
            self.mapping[g] = self.map_vars[g].get()
        save_mapping(self.mapping)
        self.running = True
        self.status_var.set("Running")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.status_var.set("Stopping...")
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")

    def save_mapping(self):
        for g in GESTURES:
            self.mapping[g] = self.map_vars[g].get()
        save_mapping(self.mapping)
        messagebox.showinfo("Saved", f"Saved mapping to {MAPPING_FILE}")

    def open_settings(self):
        # small settings window for tuning (debounce frames, swipe distance, cooldown)
        win = tk.Toplevel(self.root)
        win.title("Settings")
        ttk.Label(win, text="Debounce Frames:").grid(row=0, column=0, sticky="w")
        db = tk.IntVar(value=DEBOUNCE_FRAMES)
        ent_db = ttk.Entry(win, textvariable=db, width=6); ent_db.grid(row=0, column=1)

        ttk.Label(win, text="Swipe Window (frames):").grid(row=1, column=0, sticky="w")
        sw = tk.IntVar(value=SWIPE_WINDOW)
        ttk.Entry(win, textvariable=sw, width=6).grid(row=1, column=1)

        ttk.Label(win, text="Swipe Min Distance (norm):").grid(row=2, column=0, sticky="w")
        sdist = tk.DoubleVar(value=SWIPE_MIN_DISTANCE)
        ttk.Entry(win, textvariable=sdist, width=6).grid(row=2, column=1)

        ttk.Label(win, text="Trigger Cooldown (s):").grid(row=3, column=0, sticky="w")
        cd = tk.DoubleVar(value=TRIGGER_COOLDOWN)
        ttk.Entry(win, textvariable=cd, width=6).grid(row=3, column=1)

        def apply_and_close():
            global DEBOUNCE_FRAMES, SWIPE_WINDOW, SWIPE_MIN_DISTANCE, TRIGGER_COOLDOWN
            DEBOUNCE_FRAMES = max(1, int(db.get()))
            SWIPE_WINDOW = max(4, int(sw.get()))
            SWIPE_MIN_DISTANCE = float(sdist.get())
            TRIGGER_COOLDOWN = float(cd.get())
            self.detected_queue = deque(maxlen=DEBOUNCE_FRAMES)
            self.centroid_history = deque(maxlen=SWIPE_WINDOW)
            win.destroy()
        ttk.Button(win, text="Apply", command=apply_and_close).grid(row=4, column=0, columnspan=2, pady=6)

    def video_loop(self):
        # open capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.status_var.set("Camera error")
            self.running = False
            return
        # set reasonable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = None
            try:
                results = self.mp_hands.process(img_rgb)
            except Exception:
                results = None

            gesture_text = "No Hand"
            # default centroid
            centroid_x = None

            if results and results.multi_hand_landmarks:
                gestures_found = []
                hand_infos = []
                for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    coords3 = landmarks_to_np(hand_landmarks)  # (21,3)
                    # normalize for classifier
                    coords3n = normalize_landmarks(coords3)
                    label = handed.classification[0].label  # 'Left' or 'Right'
                    g = classify_gesture(coords3n, handedness_label=label)
                    gestures_found.append(g)
                    # compute palm centroid (use wrist (0) and middle finger mcp (9) average)
                    p0 = coords3[0, :2]; p1 = coords3[9, :2]
                    centroid = ((p0 + p1)/2.0).tolist()
                    centroid_x = centroid[0]
                    hand_infos.append((hand_landmarks, g, label, centroid))

                # choose primary gesture: highest confidence or first for now
                gesture_text = hand_infos[0][1]
                # draw landmarks
                for hl, gname, lab, cent in hand_infos:
                    mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

            # update centroid history for swipe detection
            if centroid_x is not None:
                self.centroid_history.append(centroid_x)
            else:
                self.centroid_history.append(None)

            # Detect swipe: check history valid and movement sign
            swipe_detected = None
            ch = [c for c in self.centroid_history if c is not None]
            if len(ch) >= max(3, SWIPE_WINDOW-2):
                dx = ch[-1] - ch[0]
                if dx > SWIPE_MIN_DISTANCE:
                    swipe_detected = "Swipe Right"
                elif dx < -SWIPE_MIN_DISTANCE:
                    swipe_detected = "Swipe Left"

            # Debounce: append gesture or swipe/nohand
            final_candidate = gesture_text
            if swipe_detected:
                final_candidate = swipe_detected
            if not results or not results.multi_hand_landmarks:
                final_candidate = "No Hand"
            self.detected_queue.append(final_candidate)

            # Check if last N frames are same
            if len(self.detected_queue) == self.detected_queue.maxlen and len(set(self.detected_queue)) == 1:
                final = self.detected_queue[-1]
                now = time.time()
                # ignore No Hand or Unknown for actions
                if final not in ("No Hand", "Unknown"):
                    last_t = self.last_trigger_time.get(final, 0.0)
                    if (now - last_t) > TRIGGER_COOLDOWN:
                        self.last_trigger_time[final] = now
                        action = self.mapping.get(final, "None")
                        threading.Thread(target=perform_action, args=(action,), daemon=True).start()

            # compute FPS
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if cur_time - prev_time > 0 else 0.0
            prev_time = cur_time

            # overlay text
            cv2.putText(img, f'Gesture: {self.detected_queue[-1] if self.detected_queue else "No Hand"}',
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # convert to PIL image then ImageTk
            im_pil = Image.fromarray(img_rgb)
            im_pil = im_pil.resize((800, 450))
            imgtk = ImageTk.PhotoImage(image=im_pil)

            # update UI (use after to schedule in main thread)
            self.root.after(1, self._update_frame, imgtk, int(fps), self.detected_queue[-1] if self.detected_queue else "No Hand")

        # release resources
        if self.cap:
            self.cap.release()
        self.status_var.set("Stopped")
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _update_frame(self, imgtk, fps, detected):
        # keep a reference to avoid GC
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.fps_var.set(str(fps))
        self.gesture_var.set(detected)

    def on_close(self):
        self.running = False
        time.sleep(0.2)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

# ---------- run ----------
def main():
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
