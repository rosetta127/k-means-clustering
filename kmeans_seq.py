import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from numpy.linalg import norm
import time


df = pd.read_csv('ECommerce_consumer behaviour.csv')

df.dropna(inplace=True)


class Kmeans:
    """Improved Kmeans algorithm."""

    def __init__(self, n_clusters, max_iter=100, random_state=123, init="kmeans++", n_init=10, use_median=False):
        """
        Parameters:
        - n_clusters: Number of clusters.
        - max_iter: Maximum number of iterations.
        - random_state: Seed for reproducibility.
        - init: Initialization method ('random' or 'kmeans++').
        - n_init: Number of random restarts to improve results.
        - use_median: Use median instead of mean for centroid computation to handle outliers.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.use_median = use_median

    def initialize_centroids(self, X):
        """Initialize centroids using either random or kmeans++."""
        np.random.seed(self.random_state)
        if self.init == "random":
            random_idx = np.random.permutation(X.shape[0])
            return X[random_idx[:self.n_clusters]]
        elif self.init == "kmeans++":
            centroids = [X[np.random.choice(range(X.shape[0]))]]
            for _ in range(1, self.n_clusters):
                distances = np.min(
                    np.array([norm(X - c, axis=1) for c in centroids]), axis=0
                )
                probs = distances / np.sum(distances)
                new_centroid = X[np.random.choice(range(X.shape[0]), p=probs)]
                centroids.append(new_centroid)
            return np.array(centroids)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

    def compute_centroids(self, X, labels):
        """Compute centroids using mean or median based on configuration."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] > 0:
                centroids[k, :] = (
                    np.median(cluster_points, axis=0)
                    if self.use_median
                    else np.mean(cluster_points, axis=0)
                )
        return centroids

    def compute_distance(self, X, centroids):
        """Compute the squared distance between points and centroids."""
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        """Assign points to the closest cluster."""
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        """Compute the sum of squared errors (SSE)."""
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        """Run the K-means algorithm."""
        best_sse = float("inf")
        for _ in range(self.n_init):
            # Initialize centroids
            centroids = self.initialize_centroids(X)
            for i in range(self.max_iter):
                old_centroids = centroids
                distance = self.compute_distance(X, old_centroids)
                labels = self.find_closest_cluster(distance)
                centroids = self.compute_centroids(X, labels)
                if np.allclose(old_centroids, centroids):
                    break
            sse = self.compute_sse(X, labels, centroids)
            if sse < best_sse:
                best_sse = sse
                self.centroids = centroids
                self.labels = labels
                self.error = sse

    def predict(self, X):
        """Predict the closest cluster for each data point."""
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)

    @staticmethod
    def evaluate_k(X, k_range):
        """Evaluate K using Elbow Method and Silhouette Analysis."""
        sse_scores = []
        silhouette_scores = []

        for k in k_range:
            kmeans = Kmeans(n_clusters=k, random_state=123, init="kmeans++")
            kmeans.fit(X)
            sse_scores.append(kmeans.error)
            silhouette_scores.append(silhouette_score(X, kmeans.labels))

        return sse_scores, silhouette_scores

clst_prd = pd.crosstab(df['user_id'], df['department'])
clst_prd


X_train = clst_prd.values

# Range of clusters to evaluate
no_of_clusters = range(2, 10)
inertia = []  # To store the sum of squared errors (inertia)
silhouette_scores = []  # To store the silhouette scores


# Sequential K-means timing
def run_sequential_kmeans(X_train, no_of_clusters):
    start_time = time.time()
    inertia = []
    silhouette_scores = []
    for f in no_of_clusters:
        kmeans = Kmeans(n_clusters=f, random_state=540, init="kmeans++", n_init=10, use_median=False)
        kmeans.fit(X_train)
        inertia.append(kmeans.error)
        score = silhouette_score(X_train, kmeans.labels)
        silhouette_scores.append(score)
    end_time = time.time()
    return inertia, silhouette_scores, end_time - start_time 

# Measure the sequential time
sequential_inertia, sequential_silhouette_scores, sequential_time = run_sequential_kmeans(X_train, range(2, 10))

# Write the execution time to a file
with open("sequential_execution_time.txt", "w") as file:
    file.write(f"Execution Time: {sequential_time:.4f} seconds\n")
    
results_df = pd.DataFrame({
    'Number of Clusters (k)': list(no_of_clusters),
    'Inertia': sequential_inertia,
    'Silhouette Score': sequential_silhouette_scores
})

# Round the results for better readability
results_df['Inertia'] = results_df['Inertia'].round(2)
results_df['Silhouette Score'] = results_df['Silhouette Score'].round(3)

# Print the results with proper formatting
print(f"\n{'='*50}")
print(f"Sequential K-means Clustering Results (Completed in {sequential_time:.4f} seconds)")
print(f"{'='*50}\n")
print(results_df.to_string(index=False))
print(f"\n{'='*50}")