#!/usr/bin/env python3
"""
Merged Voice Assistant with Wakeword and Stereo Vision Object Detection

Behavior:
 - The stereo vision loop (using YOLO + triangulation) runs continuously,
   showing two windows with annotated detections.
 - In parallel, the wakeword engine (using the original engine.py logic)
   continuously listens for the wakeword.
 - When the wakeword is triggered, the system records a short command.
   If the command is "all objects" (or similar), every object from the latest
   frame is announced with its depth and orientation. Otherwise, if the command
   matches a known object label (e.g. "person", "bottle", etc.), only those objects
   are announced.
 - All non-audio stereo vision processing is visible onscreen.
"""

import argparse
import threading
import time
import wave
import math
import sys

import cv2
import numpy as np
import torch
import torchaudio
import pyaudio
import whisper
import pyttsx3

from picamera2 import Picamera2
from ultralytics import YOLO
from dataset import get_featurizer

# Stereo vision helper modules (assumed valid)
import triangulation as tri
import calibration

# -----------------------------------------------------------------------------
# Global variables & Initialization
# -----------------------------------------------------------------------------

# Global list to store the latest announcement data from stereo vision.
# Each element is a dict with keys: 'name', 'depth', 'orientation'
latest_announcements = []

# To hold the frame shape (used for orientation computations)
latest_frame_shape = None

# Initialize text-to-speech engine (used for wakeword announcements)
tts_engine = pyttsx3.init('espeak')
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice',voices[17].id) #English
tts_engine.setProperty('rate', 170)

# Initialize the two PiCamera2 instances (one for right and one for left)
picam2 = Picamera2(0)
picam1 = Picamera2(1)
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
picam1.configure(picam1.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam1.start()

# Load Whisper model for transcription
whisper_model = whisper.load_model('tiny.en')

# Load the YOLO model and get class names (dictionary mapping index to name)
yolo_model = YOLO('yolo11n.pt')
class_names = yolo_model.names
print("YOLO class names:", class_names)

# -----------------------------------------------------------------------------
# Utility Function for Orientation Calculation
# -----------------------------------------------------------------------------

def get_clock_orientation(center, frame_shape):
    """
    Given the center (x,y) of a detection and the frame shape,
    compute an orientation string in a “clock” format.
    """
    h, w, _ = frame_shape
    cx, _ = center
    # Normalize horizontal offset: -1 at left edge, +1 at right edge.
    r = (cx - (w / 2)) / (w / 2)
    raw = 12 + (r * 2)  # Maps to roughly [10, 14]
    # For display, if raw > 13, subtract 12 (so 13->1, 14->2)
    if raw > 13:
        display_val = raw - 12
    else:
        display_val = raw
    return f"{display_val:.1f} o'clock", raw

# -----------------------------------------------------------------------------
# Stereo Vision Loop (Runs continuously in its own thread or main thread)
# -----------------------------------------------------------------------------

def run_stereovision_loop():
    global latest_announcements, latest_frame_shape
    while True:
        # Capture frames from both cameras
        frame_right = picam2.capture_array()
        frame_left = picam1.capture_array()

        # Undistort/rectify frames using calibration data
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
        latest_frame_shape = frame_right.shape

        start = time.time()

        # Convert frames to RGB for YOLO
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Run YOLO detection on both frames (with confidence threshold 0.5)
        results_right = yolo_model.predict(frame_right_rgb, show=False, conf=0.5)
        results_left = yolo_model.predict(frame_left_rgb, show=False, conf=0.5)

        # Convert frames back to BGR for OpenCV drawing
        frame_right = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)

        # --- STEP 1: Collect detections from right and left frames ---
        dets_right = []
        dets_left = []
        for *xyxy, conf, cls in results_right[0].boxes.data:
            cls = int(cls)
            x_min, y_min, x_max, y_max = map(int, xyxy)
            center = (x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2)
            dets_right.append({
                'cls': cls,
                'bbox': (x_min, y_min, x_max, y_max),
                'center': center,
                'conf': float(conf)
            })
            cv2.rectangle(frame_right, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        for *xyxy, conf, cls in results_left[0].boxes.data:
            cls = int(cls)
            x_min, y_min, x_max, y_max = map(int, xyxy)
            center = (x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2)
            dets_left.append({
                'cls': cls,
                'bbox': (x_min, y_min, x_max, y_max),
                'center': center,
                'conf': float(conf)
            })
            cv2.rectangle(frame_left, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # --- STEP 2: Match detections between right and left frames ---
        matches = []      # list of stereo-matched detections (with computed depth)
        matched_right = set()
        matched_left = set()

        for i, det_r in enumerate(dets_right):
            best_j = None
            best_diff = float('inf')
            for j, det_l in enumerate(dets_left):
                if det_r['cls'] == det_l['cls'] and j not in matched_left:
                    diff = abs(det_r['center'][1] - det_l['center'][1])
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j
            if best_j is not None:
                matched_right.add(i)
                matched_left.add(best_j)
                det_l = dets_left[best_j]
                # Calculate depth using the two camera centers and known stereo parameters.
                # Corrected: pass camera parameters with the proper keyword "baseline"
                depth = tri.find_depth(det_r['center'], det_l['center'], frame_right, frame_left,
                                       baseline=5.1, f=4.74, alpha=66)
                matches.append({
                    'cls': det_r['cls'],
                    'bbox_right': det_r['bbox'],
                    'bbox_left': det_l['bbox'],
                    'center_right': det_r['center'],
                    'center_left': det_l['center'],
                    'depth': depth
                })

        # --- STEP 3: Annotate matched and unmatched detections ---
        # Annotate matched detections on the right frame (including orientation)
        for match in matches:
            orientation_str, _ = get_clock_orientation(match['center_right'], frame_right.shape)
            match['orientation'] = orientation_str
            label = f"{class_names[match['cls']]}: {round(match['depth'], 1)} cm, {orientation_str}"
            x_min, y_min, x_max, y_max = match['bbox_right']
            cv2.putText(frame_right, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # For unmatched right detections, annotate with "N/A" depth and orientation
        unmatched_announcements = []
        for i, det in enumerate(dets_right):
            if i not in matched_right:
                orientation_str, _ = get_clock_orientation(det['center'], frame_right.shape)
                label = f"{class_names[det['cls']]}: N/A, {orientation_str}"
                cv2.putText(frame_right, label, (det['bbox'][0], det['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                unmatched_announcements.append({
                    'name': class_names[det['cls']],
                    'depth': "N/A",
                    'orientation': orientation_str
                })

        # (For unmatched left detections, we only annotate without depth)
        for j, det in enumerate(dets_left):
            if j not in matched_left:
                label = f"{class_names[det['cls']]}: N/A"
                cv2.putText(frame_left, label, (det['bbox'][0], det['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Build the global announcement list from both matched and unmatched detections
        announcements = []
        for match in matches:
            announcements.append({
                'name': class_names[match['cls']],
                'depth': round(match['depth'], 1),
                'orientation': match['orientation']
            })
        announcements.extend(unmatched_announcements)
        latest_announcements = announcements

        # --- STEP 4: Display FPS and show frames ---
        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Frame Right", frame_right)
        cv2.imshow("Frame Left", frame_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Wakeword Engine (Copied from engine.py)
# -----------------------------------------------------------------------------

class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening...\n")

class WakeWordEngine:
    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # run on CPU
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(8000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname

    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname, normalize=False)
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)
            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()

# -----------------------------------------------------------------------------
# Custom Action for Wakeword (Processes voice commands)
# -----------------------------------------------------------------------------

class VoiceAssistAction:
    """
    When the wakeword is detected a certain number of times consecutively,
    this action records a short command, transcribes it via Whisper,
    and then uses the latest stereo vision detections to announce objects.
    """
    def __init__(self, sensitivity=10):
        import random
        import os
        import subprocess
        import random
        from os.path import join, realpath

        self.random = random
        self.detect_in_row = 0
        self.sensitivity = sensitivity
        self.subprocess = subprocess
        folder = realpath(join(realpath(__file__), '..', 'Help'))
        self.arnold_mp3 = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if ".wav" in x
        ]

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            if self.detect_in_row == self.sensitivity:
                self.process_voice_command()
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0

    def process_voice_command(self):
        filename = self.random.choice(self.arnold_mp3)
        try:
            print("playing", filename)
            self.subprocess.check_output(['play', '-v', '.1', filename])


            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100

            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            


            print("Start recording command...")
            frames = []
            seconds = 3 # Record for 2 seconds
            for i in range(0, int(RATE / CHUNK * seconds)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("Recording stopped.")

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open("command_output.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            result = whisper_model.transcribe('command_output.wav', fp16=False)
            command = result['text'].lower().strip().replace('.', '')
            print("Command recognized:", command)

            # Process command based on detected objects from the stereo vision loop.
            global latest_announcements
            if command in ['all objects', 'all object', 'all', 'objects', 'object']:
                # Full-assist mode: announce all objects detected in the current frame.
                if not latest_announcements:
                    announcement = "No objects detected."
                else:
                    announcement_list = []
                    for ann in latest_announcements:
                        depth_str = f"{ann['depth']} centimeters" if ann['depth'] != "N/A" else "unknown depth"
                        announcement_list.append(f"{ann['name']} at {depth_str} and {ann['orientation']}")
                    announcement = "Detected objects: " + ", ".join(announcement_list)
                print("Announcing:", announcement)
                tts_engine.setProperty('rate', 90)
                tts_engine.setProperty('volume', 1)
                tts_engine.say(announcement)
                tts_engine.runAndWait()
            elif command in [name.lower() for name in class_names.values()]:
                # Keyword mode: announce only those objects matching the keyword.
                filtered = [ann for ann in latest_announcements if ann['name'].lower() == command]
                if not filtered:
                    announcement = f"No {command} detected."
                else:
                    announcement_list = []
                    for ann in filtered:
                        depth_str = f"{ann['depth']} centimeters" if ann['depth'] != "N/A" else "unknown depth"
                        announcement_list.append(f"{ann['name']} at {depth_str} at {ann['orientation']}")
                    announcement = f"Detected {command}: " + ", ".join(announcement_list)
                print("Announcing:", announcement)
                tts_engine.setProperty('rate', 90)
                tts_engine.setProperty('volume', 1)
                tts_engine.say(announcement)
                tts_engine.runAndWait()
            else:
                print("Could not understand command. Try again.")
        except Exception as e:
            print("Error processing voice command:", str(e))

# -----------------------------------------------------------------------------
# Main: Parse arguments, start wakeword engine and stereo vision loop.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice Assistant for Blind People with Stereo Vision and Wakeword")
    parser.add_argument('--model_file', type=str, required=True,
                        help="Path to the wakeword model file (optimized)")
    parser.add_argument('--sensitivity', type=int, default=10,
                        help="Wakeword sensitivity (lower value is more sensitive)")
    args = parser.parse_args()

    # Initialize and run the wakeword engine in a separate thread.
    wakeword_engine = WakeWordEngine(args.model_file)
    action = VoiceAssistAction(sensitivity=args.sensitivity)
    wakeword_thread = threading.Thread(target=wakeword_engine.run, args=(action,), daemon=True)
    wakeword_thread.start()

    # Run the stereo vision loop (this will open the display windows).
    run_stereovision_loop()
