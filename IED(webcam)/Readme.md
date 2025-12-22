* 1. I used MobileNetV2 with transfer learning and fine-tuning. 
* 2. Initially, the pretrained layers were frozen and only the classifier was trained. 
* 3. Later, the last convolutional layers were unfrozen with a low learning rate to adapt features specifically for facial emotions.
* 4. The model was then deployed for real-time webcam emotion detection using OpenCV.”

#  Real-Time Facial Emotion Detection using Transfer Learning

A **real-time facial emotion recognition system** built using **Transfer Learning (MobileNetV2)** and **TensorFlow**, capable of detecting human emotions live through a webcam.
The project focuses on **performance-aware design**, making it suitable for **CPU-based systems** like laptops.

---

##  Project Overview

Emotion detection is a challenging computer vision task due to subtle facial expressions and subjective labeling.
This project leverages a **pre-trained deep learning model (MobileNetV2)** to extract high-level facial features and classifies them into different emotional categories.

The system:

* Trains on a labeled facial emotion dataset
* Uses **fine-tuning** to improve accuracy
* Performs **real-time inference** using a webcam

---

##  Features

* ✅ Transfer Learning with **MobileNetV2**
* ✅ Two-stage training (Frozen base + Fine-tuning)
* ✅ Real-time webcam emotion prediction
* ✅ Data augmentation for better generalization
* ✅ Performance-optimized for CPU environments
* ✅ Accuracy & loss visualization
* ✅ Clean, modular, single-file implementation

---

##  Emotions Detected

The model classifies facial expressions into **7 categories**:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

##  Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **MobileNetV2 (ImageNet pretrained)**

---

##  Project Structure

```
emotion_detection/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── neutral/
│   │   ├── fear/
│   │   ├── disgust/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── happy/
│       ├── sad/
│       ├── neutral/
│       ├── fear/
│       ├── disgust/
│       └── surprise/
│
├── IED.py
├── requirements.txt
└── README.md
```

---

##  Dataset

* Facial emotion image dataset (e.g. **FER-2013** or similar)
* Images organized by emotion labels
* Balanced train/test split recommended

>  Emotion recognition datasets are noisy and subjective.
> Even state-of-the-art models typically achieve **60–70% accuracy**.

---

##  Installation & Setup

### 1.  Clone the Repository

```bash
git clone https://github.com/vidushi-8465/emotion-detection.git
cd emotion-detection
```

---

### 2️. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
tensorflow
opencv-python
numpy
matplotlib
```

---

## 4. Running the Project

```bash
python IED.py
```

### Execution Flow:

1. Dataset is loaded and augmented
2. Model is trained using transfer learning
3. Fine-tuning improves feature adaptation
4. Model is saved locally
5. Webcam opens for real-time emotion detection
6. Press **Q** to exit webcam window

---

##  Training Results

* Gradual improvement in training and validation accuracy
* Decreasing loss curves indicate stable learning
* Final validation accuracy ~**45–60%** (dataset dependent)

> This accuracy is **realistic** for emotion recognition and reflects the complexity of the task.

---

##  Why Accuracy is Limited

* Facial emotions are subtle and subjective
* Overlapping expressions (sad vs neutral, fear vs surprise)
* Dataset noise and lighting variations
* CPU-based training constraints

Despite this, the model generalizes well and performs reliably in real-time scenarios.

---

##  Key Concepts Used

* **Transfer Learning**
* **Fine-Tuning**
* **Data Augmentation**
* **Softmax Classification**
* **Real-Time Computer Vision**
* **Performance Optimization**

---

##  Future Improvements

* 🔹 Face detection before emotion prediction
* 🔹 GPU-based training for higher accuracy
* 🔹 FPS optimization for real-time deployment
* 🔹 Web or mobile deployment
* 🔹 Emotion tracking over time

---

##  Conclusion

This project demonstrates a **practical, real-world application of deep learning**, combining model optimization, transfer learning, and real-time inference.
It highlights both **machine learning understanding** and **system-level thinking**, making it suitable for academic evaluation and technical interviews.

---
