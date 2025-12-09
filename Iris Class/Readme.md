--- 
# Day 1 – Iris Flower Classification

This project builds a machine learning model that can identify the type of an Iris flower (Setosa, Versicolor, Virginica) using four measurements:
**sepal length, sepal width, petal length, and petal width.**
---
## 📌 What This Project Does
* Loads the Iris dataset
* Trains 4 ML algorithms
* Compares their accuracy
* Picks the **best model (SVM)**
* Saves it as `iris_best_model.pkl`
---
## 🧠 Algorithms Used (Very Simple Explanation)
### **1. Logistic Regression**
Draws simple straight boundaries between flower types.
### **2. KNN (K-Nearest Neighbours)**
Looks at nearby flowers and predicts based on what most neighbours are.
### **3. SVM (Support Vector Machine)**
Builds the strongest possible wall between different flower groups.
**Best performer in this project.**
### **4. Decision Tree**
Asks a series of yes/no questions (like a flowchart) to reach a prediction.
---

## 🏁 Output

* SVM achieved the highest accuracy (~97%).
* Model saved for future prediction.
---

## 📚 What I Learned
* Basics of ML model training
* Comparing algorithms
* Evaluating accuracy & confusion matrices
* Saving trained models
---
