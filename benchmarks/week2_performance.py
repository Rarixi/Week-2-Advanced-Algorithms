import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd

from src.sorting.basic_sorts import (
    bubble_sort,
    selection_sort,
    insertion_sort
)
from src.sorting.merge_sort import merge_sort
from src.sorting.quick_sort import quick_sort


# Input sizes required by the assignment
SIZES = [100, 500, 1000, 5000, 10000, 50000]


# All sorting algorithms being tested
ALGORITHMS = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort
}


# Create the results folder if it does not already exist
os.makedirs("benchmarks/results", exist_ok=True)


# -----------------------------
# DATA GENERATION FUNCTIONS
# -----------------------------

def generate_random(size):
    """
    Create a list of random numbers.
    """
    return [random.randint(0, size) for _ in range(size)]


def generate_sorted(size):
    """
    Create an already sorted list.
    """
    return list(range(size))


def generate_reverse_sorted(size):
    """
    Create a list sorted in reverse order.
    """
    return list(range(size, 0, -1))


def generate_nearly_sorted(size):
    """
    Create a list that is approximately 95% sorted.
    """
    data = list(range(size))

    swaps = max(1, int(size * 0.05))

    for _ in range(swaps):
        index1 = random.randint(0, size - 1)
        index2 = random.randint(0, size - 1)

        data[index1], data[index2] = data[index2], data[index1]

    return data


def generate_many_duplicates(size):
    """
    Create data using only 10 unique values.
    """
    return [random.randint(0, 9) for _ in range(size)]


def generate_few_unique(size):
    """
    Create data using only 3 unique values.
    """
    return [random.randint(0, 2) for _ in range(size)]


DATA_TYPES = {
    "Random": generate_random,
    "Sorted": generate_sorted,
    "Reverse Sorted": generate_reverse_sorted,
    "Nearly Sorted": generate_nearly_sorted,
    "Many Duplicates": generate_many_duplicates,
    "Few Unique": generate_few_unique
}


# This list will store every benchmark result
results = []


# -----------------------------
# RUN BENCHMARKS
# -----------------------------

for data_type, generator in DATA_TYPES.items():

    print(f"\n==============================")
    print(f"Testing: {data_type}")
    print(f"==============================")

    for size in SIZES:

        print(f"\nInput size: {size}")

        # Create one dataset
        data = generator(size)

        # Give every algorithm the same data
        for name, algorithm in ALGORITHMS.items():

            test_data = data.copy()

            print(f"Running {name}...")

            start = time.perf_counter()

            algorithm(test_data)

            end = time.perf_counter()

            elapsed = end - start

            print(f"{name}: {elapsed:.6f} seconds")

            # Save the result
            results.append({
                "algorithm": name,
                "data_type": data_type,
                "size": size,
                "time": elapsed
            })


# -----------------------------
# SAVE RESULTS TO CSV
# -----------------------------

df = pd.DataFrame(results)

df.to_csv(
    "benchmarks/results/comparison_table.csv",
    index=False
)

print("\nSaved comparison table:")
print("benchmarks/results/comparison_table.csv")


# -----------------------------
# CREATE PERFORMANCE PLOTS
# -----------------------------

PLOT_FILES = {
    "Random": "random_data.png",
    "Sorted": "sorted_data.png",
    "Reverse Sorted": "reverse_data.png",
    "Nearly Sorted": "nearly_sorted.png",
    "Many Duplicates": "many_duplicates.png",
    "Few Unique": "few_unique.png"
}


for data_type, filename in PLOT_FILES.items():

    plt.figure()

    subset = df[df["data_type"] == data_type]

    for algorithm in subset["algorithm"].unique():

        algorithm_data = subset[
            subset["algorithm"] == algorithm
        ]

        plt.plot(
            algorithm_data["size"],
            algorithm_data["time"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Input Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Sorting Performance - {data_type}")
    plt.legend()
    plt.grid(True)

    output_path = f"benchmarks/results/{filename}"

    plt.savefig(output_path)

    plt.close()

    print(f"Saved plot: {output_path}")


print("\nBenchmark complete.")
