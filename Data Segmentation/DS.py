import pandas as pd                                #Used to handle the dat in the form of tables
import numpy as np                                 #Used for mathematical calculations
import matplotlib.pyplot as plt                    #Used to plot graphs
from sklearn.cluster import KMeans                 #we are using K-means algo : It is a clustering algorithm
from sklearn.preprocessing import StandardScaler   #Standardscaler is used to compare the numbers

DATA_PATH = "data/customer.csv"                    # Path to where the csv file of the Mall customers is saved

# We are now reading the file into the dataframe
def load_data(path):                               
    return pd.read_csv(path)

#It is used for preprocessing that is selecting only the required columns 
def preprocess(df):
    
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # We scale the values so that income and spendings are treated equally preventing any biasness due to different ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # This is used to return both original and scaled data
    return X, X_scaled

# This is the Elbow method used to find the best K
def find_optimal_k(X_scaled):

    #empty list to store errors
    wcss = []
    # this is to train k-means for different K values ie. from 1 to 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        # To store WCSS i.e within sum of Squares
        wcss.append(kmeans.inertia_)
    # This is to Plot the Graph  to "find the elbow point"
    plt.plot(range(1, 11), wcss)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()

#This is to train the K-means model
def train_kmeans(X_scaled, k=5):
    # Create Kmeans model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # Fit a model and assign each customer to a cluster
    labels = kmeans.fit_predict(X_scaled)
    # Return kmeans model and labels
    return kmeans, labels

#to start the plotting
def visualize_clusters(X, labels):
   
   #this is to plot customers in @-D
    plt.scatter(
        X.iloc[:, 0], # X- axis --> Income
        X.iloc[:, 1], # Y- axim --> Spendings
        c=labels      # c= colour --> cluster Number
    )
    
    # this is to plot the names or labels at x -axis , Y- axis and The title
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")

    plt.title("Customer Segmentation")
    plt.show()


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    
    X, X_scaled = preprocess(df)
    
    find_optimal_k(X_scaled)
    kmeans, labels = train_kmeans(X_scaled, k=5)
    df['Cluster'] = labels
    visualize_clusters(X, labels)
    print(df.head())
