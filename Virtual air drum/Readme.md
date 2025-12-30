---

##  Day 6 (Project 3): Virtual Air Drum & Piano using Computer Vision

This project is a **real-time virtual musical instrument** that allows users to **play drums and piano in the air** using only their **hand gestures and a webcam**.

It uses **computer vision and hand-tracking** to detect finger positions and triggers musical sounds when the user interacts with virtual drum pads and piano keys displayed on the screen.

---

##  Overview

Traditional musical instruments require physical hardware.
This project removes that barrier by allowing users to **play music virtually** using hand movements.

The application works by:

• Capturing live webcam feed  
• Detecting hand landmarks using MediaPipe  
• Tracking the index finger position  
• Mapping finger movement to virtual instruments  
• Playing corresponding sounds in real time  

---

##  Tech Stack Used

### Computer Vision & Logic

• Python  
• OpenCV – video capture and rendering  
• MediaPipe – real-time hand tracking  

### Audio Processing

• Pygame – sound playback and mixing  

### Libraries

• `opencv-python` – webcam input and display  
• `mediapipe` – hand landmark detection  
• `numpy` – coordinate calculations  
• `pygame` – low-latency audio playback  

---

##  Project Structure

```md
---
##  Project Structure
```

Virtual-air-drum/
│
├── vad.py                 # Main Python file (application logic)
├── requirements.txt       # Project dependencies
│
├── sounds/                # Audio files
│   ├── drum1.wav
│   ├── drum2.wav
│   ├── piano1.wav
│   └── piano2.wav
│
├── images/                # Instrument UI images (PNG with transparency)
│   ├── drum_left.png
│   ├── drum_right.png
│   ├── piano_left.png
│   └── piano_right.png
│
├── .gitignore              # Git ignore file (excludes venv, cache, etc.)
└── README.md               # Project documentation

````
---
##  How It Works (Flow)

1. Webcam captures live video feed
2. MediaPipe detects hand landmarks
3. Index finger tip position is tracked
4. Screen is divided into interaction zones:
   - Top → Drum pads
   - Bottom → Piano keys
5. Instrument images are overlaid on the webcam feed
6. When the finger enters a zone:
   - Corresponding sound is played using Pygame
7. Output is rendered in real time with minimal latency
---
##  Interaction Zones

| Screen Area | Instrument |
|------------|------------|
| Top Left | Drum 1 |
| Top Right | Drum 2 |
| Bottom Left | Piano Key 1 |
| Bottom Right | Piano Key 2 |

---
##  Features

• Real-time hand tracking  
• Air-based drum and piano interaction  
• Low-latency sound playback  
• Visual instrument overlays instead of plain rectangles  
• Single-file main logic  
• No internet or API required  

---
##  Limitations

1. Requires a webcam
2. Sound may repeat without finger tap detection
3. Lighting conditions affect tracking accuracy
4. Limited number of instrument keys
5. Desktop-only (not browser-based)

---
##  Key Learnings from this Project

• Real-time hand tracking with MediaPipe  
• Integrating computer vision with audio systems  
• Coordinate mapping and interaction zones  
• Overlaying transparent PNGs on live video  
• Managing performance in real-time applications  
• Structuring computer vision projects professionally  

---

##  How to Run
```bash
pip install -r requirements.txt
python vad.py
````
--- 


