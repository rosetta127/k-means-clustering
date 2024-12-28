import time
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

# Function to measure the parallel execution time (using MPI)
def measure_parallel_time(num_processes):
    # Run the parallel K-means using subprocess to invoke parallel.py with MPI
    command = f"mpiexec -n {num_processes} python kmeans_par.py"
    subprocess.run(command, shell=True, check=True)  # This will run the parallel code
    
    # Read the parallel time from the file
    with open("execution_time.txt", "r") as f:
        line = f.read().strip()  # Read the line and strip any leading/trailing whitespace
        # Extract the numeric value from the string
        parallel_time = float(line.split(":")[1].split()[0])  # Get the number after the colon and before the "seconds"
    
    return parallel_time

# Measure the sequential execution time
def measure_sequential_time():
    with open("sequential_execution_time.txt", "r") as f:
        line = f.read().strip()  # Read the line and strip any leading/trailing whitespace
        # Extract the numeric value from the string
        sequential_time = float(line.split(":")[1].split()[0])  # Get the number after the colon and before "seconds"
    return sequential_time

# Sequential time (precomputed)
sequential_time = measure_sequential_time()

# Range of processes to test
processes_range = [2, 3, 4, 5, 6, 7]
parallel_times = []
speedups = []

# Measure parallel time for each process count and calculate speedup
for num_processes in processes_range:
    parallel_time = measure_parallel_time(num_processes)
    parallel_times.append(parallel_time)
    speedups.append(sequential_time / parallel_time)  # Calculate speedup
    print(f"Processes: {num_processes}, Parallel Time: {parallel_time:.4f}, Speedup: {speedups[-1]:.2f}")

# Plot Execution Time vs Number of Processes
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(processes_range, parallel_times, marker='o', linestyle='-', color='r', label='Parallel Time')
plt.title('Execution Time vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.legend()

# Plot Speedup vs Number of Processes
plt.subplot(2, 1, 2)
plt.plot(processes_range, speedups, marker='o', linestyle='-', color='b', label='Speedup')
plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()

# Save the plots to files
plt.tight_layout()
plt.savefig('execution_time_and_speedup_plots.png')

# Optionally, show the plots
plt.show()
