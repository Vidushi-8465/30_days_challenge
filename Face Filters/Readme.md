# Face Filter using OpenCV (Mask Overlay)

A **real-time computer vision project** that detects human faces using **Haar Cascade classifiers** and applies a **transparent mask filter** on detected faces using OpenCV.

---

##  Approach Used

* 1. Webcam frames are captured in real time using OpenCV.
* 2. Faces are detected using a pretrained **Haar Cascade** model.
* 3. A transparent PNG mask is resized based on face dimensions.
* 4. Alpha blending is used to overlay the mask naturally on the face.
* 5. The filtered video is displayed in real time.

---

##  Features

* Real-time face detection
* Transparent mask overlay
* Works with webcam
* Lightweight and CPU-efficient
* Modular and easy-to-extend codebase

---

##  Tech Stack

* **Python**
* **OpenCV**
* **NumPy**

---

##  Project Structure

```
face_filter_opencv/
│
├── src/
│   ├── main.py
│   ├── face_detector.py
│   └── mask_overlay.py
│
├── assets/
│   └── mask.png
│
├── requirements.txt
└── README.md
```

---

##  How to Run

```bash
pip install -r requirements.txt
cd src
python main.py
```

---

##  Conclusion

This project demonstrates **real-time face detection and image overlay using OpenCV**, combining classical computer vision techniques with pixel-level alpha blending for visual effects.

---

##  Possible Extensions

* Multiple filters (glasses, hats, emojis)
* Face landmark-based alignment
* Snapchat-style filter switching
* Mobile or web deployment

---
