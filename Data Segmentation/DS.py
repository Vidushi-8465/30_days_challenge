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
    """
    Select features and scale them
    """
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # We scale the values so that income and spendings are treated equally preventing any biasness due to different ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled
def find_optimal_k(X_scaled):
    """
    Use Elbow Method to find best K
    """
    wcss = []
    
    for k in range(1, 11):

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        kmeans.fit(X_scaled)

        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()

def train_kmeans(X_scaled, k=5):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X_scaled)

    return kmeans, labels

def visualize_clusters(X, labels):

    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels
    )

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
