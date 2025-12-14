### Day 5: Customer Segmentation using K-Means 

This project segments customers into different groups using **K-Means Clustering** based on their purchasing behavior.

## Overview: 

Customer segmentation helps businesses understand different types of customers.  
In this project, customers are grouped using:
- Annual Income
- Spending Score

Similar customers fall into the same cluster.

### Dataset used: 
We have used the most common mall data

### Algorithms used : 

* 1. :K-Means Clustering
      * What is it?

> An unsupervised learning algorithm that groups data into K clusters.
> No labels. The algorithm finds patterns on its own.
   
      * Why do we use K-Means? 
> K-Means is simple, fast, scalable, and works well when clusters are spherical and well-separated.
      * How it works (Step-by-step): 

>Choose K (number of clusters)
>Randomly place K centroids
>Assign each point to the nearest centroid
>Recalculate centroids
>Repeat until centroids stop moving

       * Why scaling is important?

>K-Means uses distance
>Different feature ranges = wrong clusters
>Hence we used: StandardScaler()

* 2. Elbow Method
>Used to find optimal K
>Plot K vs WCSS
>The point where reduction slows → best K
>Looks like an “elbow”

* 3. StandardScaler :
   Formula: 
>Z = (X - mean) / standard deviation
>This ensures that mean =0 ; Standard Deviation = 1

