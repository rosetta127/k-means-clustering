import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from numpy.linalg import norm
import time
from mpi4py import MPI

# Reading the data
df = pd.read_csv('ECommerce_consumer behaviour.csv')
df.dropna(inplace=True)

class Kmeans:
    """Kmeans algorithm."""
    def __init__(self, n_clusters, max_iter=100, random_state=123, init="kmeans++", n_init=10, use_median=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.use_median = use_median

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        if self.init == "random":
            random_idx = np.random.permutation(X.shape[0])
            return X[random_idx[:self.n_clusters]]
        elif self.init == "kmeans++":
            centroids = [X[np.random.choice(range(X.shape[0]))]]
            for _ in range(1, self.n_clusters):
                distances = np.min(np.array([norm(X - c, axis=1) for c in centroids]), axis=0)
                probs = distances / np.sum(distances)
                new_centroid = X[np.random.choice(range(X.shape[0]), p=probs)]
                centroids.append(new_centroid)
            return np.array(centroids)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

    def compute_centroids(self, X, labels):
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
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        best_sse = float("inf")
        for _ in range(self.n_init):
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
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)

    @staticmethod
    def evaluate_k(X, k_range):
        sse_scores = []
        silhouette_scores = []
        for k in k_range:
            kmeans = Kmeans(n_clusters=k, random_state=123, init="kmeans++")
            kmeans.fit(X)
            sse_scores.append(kmeans.error)
            silhouette_scores.append(silhouette_score(X, kmeans.labels))
        return sse_scores, silhouette_scores

# Reading the data
clst_prd = pd.crosstab(df['user_id'], df['department'])
X_train = clst_prd.values

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Range of clusters to evaluate
no_of_clusters = range(2, 10)

# Sequential K-means function to run in parallel
def run_parallel_kmeans(X_train, no_of_clusters):
    # Scatter the cluster ranges to different processes
    clusters_per_process = len(no_of_clusters) // size
    start_idx = rank * clusters_per_process
    end_idx = (rank + 1) * clusters_per_process if rank != size - 1 else len(no_of_clusters)

    local_inertia = []
    local_silhouette_scores = []
    
    for k in no_of_clusters[start_idx:end_idx]:
        kmeans = Kmeans(n_clusters=k, random_state=540, init="kmeans++", n_init=10, use_median=False)
        kmeans.fit(X_train)
        local_inertia.append(kmeans.error)
        score = silhouette_score(X_train, kmeans.labels)
        local_silhouette_scores.append(score)

    return local_inertia, local_silhouette_scores

# Measure the parallel time
start_time = time.time()

# Run the parallel computation
local_inertia, local_silhouette_scores = run_parallel_kmeans(X_train, no_of_clusters)

# Gather the results from all processes to the root process (rank 0)
all_inertia = comm.gather(local_inertia, root=0)
all_silhouette_scores = comm.gather(local_silhouette_scores, root=0)

# If the rank is 0, combine and print the results
if rank == 0:
   # Flatten the lists received from all processes
   inertia = [item for sublist in all_inertia for item in sublist]
   silhouette_scores = [item for sublist in all_silhouette_scores for item in sublist]
   cluster_range = list(no_of_clusters)  # Ensure the range of k is preserved
   
   # Measure the time taken for parallel execution
   end_time = time.time()
   
   # Write the execution time to a file
   with open("execution_time.txt", "w") as file:
       file.write(f"Execution Time: {end_time - start_time:.4f} seconds\n")
   
   # Create a DataFrame for a cleaner output
   results_df = pd.DataFrame({
       'Number of Clusters (k)': cluster_range,
       'Inertia': inertia,
       'Silhouette Score': silhouette_scores
   })
   
   # Print the results in a nice format
   print(f"\n{'='*50}")
   print(f"Parallel K-means Clustering Results (Completed in {end_time - start_time:.4f} seconds)")
   print(f"{'='*50}\n")
   print(results_df.to_string(index=False))
   print(f"\n{'='*50}")
