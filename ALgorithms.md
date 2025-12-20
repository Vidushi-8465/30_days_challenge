Amazing question 👏
This is exactly how **real ML understanding** is built — not just coding, but knowing *what* you used and *why*.

Below is a **complete, structured breakdown of ALL algorithms you’ve used so far (Day 1 → Day 7)**, explained **from basics**, with **code snippets** and **intuition**.
I’ll explain like a **beginner**, but with **technical correctness**.

---

# 📚 COMPLETE LIST OF ALGORITHMS YOU USED (30-Day Challenge)

---

## 🔹 1. Train–Test Split

**(Used in almost every project)**

### What is it?

We split our data into:

* **Training data** → model learns
* **Testing data** → model is evaluated

### Why?

If we test on training data → cheating ❌
We want to know how well the model performs on **new data**.

### Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🔹 2. Logistic Regression (Day 1 – Iris)

### What is it?

A **classification algorithm** that predicts probabilities for classes.

Despite the name, it’s used for **classification**, not regression.

### How it works (simple):

* Draws a line (or curve) between classes
* Uses **sigmoid function** to map output between 0–1

### Code

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 🔹 3. K-Nearest Neighbors (KNN) (Day 1)

### What is it?

A **lazy learning algorithm**.

It doesn’t learn — it **remembers** all data.

### How it works:

1. Choose K (e.g., 5)
2. Find K closest points
3. Majority class wins

### Code

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

---

## 🔹 4. Support Vector Machine (SVM) (Day 1)

### What is it?

A powerful classifier that finds the **best boundary** between classes.

### Key idea:

> Maximize the distance between classes (maximum margin)

### Why it performed best in Iris:

* Iris is well-separated
* SVM handles this perfectly

### Code

```python
from sklearn.svm import SVC

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
```

---

## 🔹 5. Decision Tree (Day 1)

### What is it?

A **flowchart-like** model.

### How it works:

* Ask questions like:

  * Is petal length < 2.5?
  * Yes → left, No → right

### Pros:

* Easy to understand
* Handles non-linear data

### Cons:

* Can overfit

### Code

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
```

---

## 🔹 6. Cross Validation (Day 1)

### What is it?

Instead of one train-test split:

* Train & test multiple times
* Take average accuracy

### Why?

More reliable performance estimate.

### Code

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```

---

## 🔹 7. Linear Regression (Day 2 – House Price)

### What is it?

A regression algorithm that finds a **straight line** best fitting the data.

### Formula:

```
y = mx + c
```

### Used when:

* Relationship is linear
* Output is continuous

### Code

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
```

---

## 🔹 8. Random Forest (Day 2)

### What is it?

An **ensemble algorithm** made of many decision trees.

### Key idea:

> Many weak models → one strong model

### Why better than single tree?

* Reduces overfitting
* More accurate

### Code

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train, y_train)
```

---

## 🔹 9. Evaluation Metrics (Regression)

### MAE

Average absolute error

```python
mean_absolute_error(y_true, y_pred)
```

### MSE

Squares errors (penalizes large errors)

```python
mean_squared_error(y_true, y_pred)
```

### RMSE

Error in original units

```python
np.sqrt(mse)
```

### R² Score

How well model explains data

```python
r2_score(y_true, y_pred)
```

---

## 🔹 10. Cosine Similarity (Day 4 – Movie Recommendation)

### What is it?

Measures **angle** between vectors, not distance.

### Used for:

* Text similarity
* Recommendation systems

### Formula:

```
cos(θ) = (A · B) / (|A| |B|)
```

### Code

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)
```

---

## 🔹 11. TF-IDF Vectorizer (NLP projects)

### What is it?

Converts text → numbers.

### Meaning:

* TF → word frequency
* IDF → importance of word

### Why?

Words like *the, is* are ignored.

### Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(text)
```

---

## 🔹 12. K-Means Clustering (Day 5)

### What is it?

An **unsupervised algorithm** that groups similar data.

### Steps:

1. Choose K
2. Assign points
3. Update centroids
4. Repeat

### Code

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
```

---

## 🔹 13. Elbow Method (Day 5)

### What is it?

Helps choose optimal K.

### Code

```python
kmeans.inertia_
```

Plot inertia vs K.

---

## 🔹 14. StandardScaler

### What is it?

Scales data so:

* Mean = 0
* Std = 1

### Why important?

Distance-based algorithms need scaling.

### Code

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🔹 15. Isolation Forest (Day 6 – Fraud Detection)

### What is it?

An **anomaly detection algorithm**.

### Idea:
> Anomalies are easier to isolate.

### Code
```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01)
model.fit(X)
```
---

##  16. Naive Bayes (Day 6 – Sentiment Analysis)

### What is it?
A probabilistic classifier based on Bayes theorem.

### Assumption:
Features are independent (naive).

### Best for:
* Text classification
* Spam detection
* Sentiment analysis

### Code
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
```
---

## 17. Classification Metrics
### Accuracy
```python
accuracy_score(y_test, y_pred)
```
### Precision / Recall / F1
classification_report(y_test, y_pred)
```
---
