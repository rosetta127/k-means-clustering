# K-Means Clustering: Sequential and Parallel Implementations

This repository contains both sequential and parallel implementations of the K-Means clustering algorithm. The implementation uses the MPI4Py library to enable parallel processing and includes utilities for measuring speedup with respect to the number of processes used.

---

## Sequential version
1. K-means Algorithm Implementation:

* The clustering algorithm (fit method in Kmeans class) runs sequentially for each value of k.
* This includes centroid initialization, assignment of data points to clusters, updating centroids, and calculating SSE.

---
## Parallel version

The part of the algorithm that was parallelized is the evaluation of multiple k values. The range of k values is divided among the available MPI processes. Each process independently computes the inertia (SSE) and silhouette scores for its assigned k values using the K-Means algorithm. This approach significantly reduces the overall computation time compared to evaluating all k values sequentially.

## Requirements

* Python 

* Required libraries: pandas, numpy, matplotlib, mpi4py, scikit-learn

* MPI installation
  
* download the dataset via this link https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023/data

## Data

* Input Data: The dataset ECommerce_consumer behaviour.csv should be placed in the same directory as the code files. 

* The dataset is preprocessed to create a cross-tabulated matrix of user activity by department.

## Files

* kmeans_seq.py: Contains the sequential implementation.

* kmeans_par.py: Contains the parallel implementation.

* calculate_speedup.py: Measures execution time and computes speedup.


## Instructions to Reproduce Results

### Data Preparation
- **Input Data**: Place the dataset (`ECommerce_consumer_behaviour.csv`) in the root directory of this repository.
  - Format: CSV file with no missing values. The `user_id` and `department` columns are used to generate the clustering data matrix.

### Running the Sequential Implementation

1. Run the script `kmeans_seq.py`:
   ```bash
   python kmeans_seq.py
   ```
2. Outputs:
   - Cluster evaluation metrics (inertia and silhouette score) for values of `k` in the range [2, 10].
   - Execution time recorded in `sequential_execution_time.txt`.

### Running the Parallel Implementation
1. Install MPI4Py.
2. Run the parallel script `kmeans_par.py` with the desired number of processes:
   ```bash
   mpiexec -n <num_processes> python kmeans_par.py
   ```
3. Outputs:
   - Cluster evaluation metrics aggregated across processes.
   - Execution time recorded in `execution_time.txt`.

### Speedup Calculation
1. Run the script `calculate_speedup.py` to compute and visualize speedup:
   ```bash
   python calculate_speedup.py
   ```
2. Output:
   - A plot showing execution time and speedup vs. number of processes.
   - Saved as `execution_time_and_speedup_plots.png`.

---


## Speedup Calculation
The speedup achieved by the parallel implementation is calculated as:

**Speedup = (Time for Sequential Implementation) / (Time for Parallel Implementation)**

### Results
- Sequential execution time is recorded in `sequential_execution_time.txt`.
- Parallel execution time for different numbers of processes is recorded in `execution_time.txt`.
- The `calculate_speedup.py` script generates plots illustrating:
  - Execution Time vs. Number of Processes
  - Speedup vs. Number of Processes

These plots help visualize the efficiency gains achieved through parallelization.

---

![Image](https://github.com/user-attachments/assets/d932e28f-036e-4f01-be6d-ff7dfeb24239)
