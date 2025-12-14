### Day 6 (Project 1): Credit Card Fraud Detection using Anomaly Detection

This project detects fraudulent credit card transactions using **Anomaly Detection**.

## Overview:

Fraud transactions are rare and different from normal transactions.
Instead of learning fraud patterns, the model learns normal behavior and
flags unusual transactions as fraud.

### Dataset used:
Credit Card Transactions Dataset (highly imbalanced)

### Algorithms used:

* 1. Isolation Forest
   * What is it?
> An anomaly detection algorithm that isolates unusual data points.
> It works without needing labeled fraud data.

   * Why do we use Isolation Forest?
> Fraud data is rare.
> Isolation Forest works well on imbalanced datasets.
> It is fast and scalable.

   * How it works:
> Randomly splits data
> Anomalies get isolated faster
> Shorter paths = anomalies

* 2. StandardScaler
> Scales features so distance-based learning works correctly
