---

## Colour Detection Wheel using Dominant Colours (OpenCV)

A **computer vision project** that extracts **dominant colours** from an image using **K-Means clustering** and visualizes them as a **circular colour wheel** using OpenCV.

---

## Approach Used

* 1. The input image is reshaped into pixel-level RGB data.
* 2. **K-Means clustering** is applied to group similar colors.
* 3. Cluster centers represent the dominant colours in the image.
* 4. Dominant colours are sorted by frequency.
* 5. A circular colour wheel is generated using OpenCV for visualization.

---

## Features

* Unsupervised dominant colour extraction
* No labeled data required
* Works with any image
* Clean modular code structure
* Lightweight and CPU-efficient

---

## 🛠 Tech Stack

* **Python**
* **OpenCV**
* **NumPy**
* **Scikit-learn**

---

## Project Structure

```
Colour Detection Wheel/
│
├── src/
│   ├── main.py
│   ├── colour_extractor.py
│   └── wheel_generator.py
│
├── Images/
│   └── img1.jpg
│   └── img3.jpg
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
cd src
python main.py
```

---

## Applications

* Colour palette extraction
* Image analysis
* UI/UX design support
* Computer vision learning projects

---

## Conclusion

This project demonstrates the practical use of **K-Means clustering in computer vision**, combining algorithmic colour analysis with intuitive visual representation.

---
