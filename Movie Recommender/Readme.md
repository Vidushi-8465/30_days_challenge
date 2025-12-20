---
## Day 4 :Movie Recommendation System (Content-Based)
A simple **content-based movie recommender** built using **TF-IDF** and **Cosine Similarity**.
The system suggests movies similar to a selected movie based on textual features like **genres and overview**.
---

## 🔹 Features
* Recommends movies similar to a given movie
* Uses movie metadata (genres + overview)
* Content-based filtering (no user history required)
* Fast and lightweight implementation
---

##  Tech Stack & Libraries
* **Python 3.x**
* **Pandas** → Data handling
* **scikit-learn**

  * `TfidfVectorizer`
  * `cosine_similarity`
---

## Folder Structure
```
Movie_Recommender/
│
├─ movie_recommender.py   # Main script
├─ movies.csv             # Dataset
└─ README.md              # Project documentation
```
---

##  Basic Flow

1. Load the movie dataset
2. Combine text features (genres + overview)
3. Convert text into TF-IDF vectors
4. Compute cosine similarity between movies
5. Recommend top similar movies

---

##  Algorithms & Techniques Used

* **TF-IDF Vectorization** – Converts movie text into numerical vectors
* **Cosine Similarity** – Measures similarity between movies
* **Content-Based Filtering** – Recommends based on movie content
---

##  Output
```
Movie selected: Inception

Recommended Movies:
1. Interstellar
2. The Prestige
3. The Matrix
```
---