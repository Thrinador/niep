# NIEP Exploration Toolkit

## 📝 Overview

This repository provides a collection of research tools for the numerical exploration of the **Nonnegative Inverse Eigenvalue Problem (NIEP)** and its primary sub-problems:
* **NIEP**: The general Nonnegative Inverse Eigenvalue Problem for stochastic matrices.
* **SNIEP**: The Symmetric Nonnegative Inverse Eigenvalue Problem, assuming stochastic matrices.
* **sub-SNIEP**: The Symmetric NIEP for sub-stochastic matrices, scaled such that the Perron root is 1.

The core of this toolkit is an optimization-based approach to numerically map the boundaries of the realizable region of coefficients for the characteristic polynomial. By identifying matrices that form this boundary, the toolkit calculates their corresponding eigenvalues (spectra), thereby constructing the boundary of the realizable spectra space.

---

## ⚙️ Core Components

The toolkit consists of two main scripts and a central configuration file that drives all operations.

1.  **`optimizer_main.py`**: This is the primary script for generating data. It systematically finds the boundaries of the coefficient space ($E_k$, the sum of k-by-k principal minors) by performing a series of constrained optimizations. Once boundary points are found, it computes their eigenvalues, saves the resulting matrix, coefficients, and spectra to a JSON file, and generates plots of the solution space.

2.  **`extreme_points.py`**: This script is a utility for analyzing the geometry of the solution space. It takes a set of known or guessed "extreme points" and a larger dataset (e.g., from the optimizer). It then performs a convex hull analysis to:
    * Identify which of the initial points are redundant (can be formed by a convex combination of others).
    * Find data points from the larger set that lie outside the current convex hull.
    * Report the equations of the exposed facets of the hull.
    This is invaluable for iteratively building a complete description of the solution space boundary.

3.  **`config.toml`**: This file is the central control panel for the entire toolkit. All parameters—from matrix size and type to optimizer tolerances and file paths—are defined here, allowing for repeatable and well-defined experiments without altering the source code.

---

## 🔧 Setup and Installation

All setup is done in the `config.toml` file. Key sections include:

* **`[global_data]`**:
    * `n`: The size of the matrix (e.g., 4, 5, 6).
    * `matrix_type`: The problem to solve (`niep`, `sniep`, `sub_sniep`).
    * `funcs_to_optimize`: A list of the coefficients ($E_k$) to optimize.
    * `points_dim`: The number of sample points for each optimization axis.

* **`[optimize_data]`**:
    * Tolerances (`tol_nlc`, `tol_slsqp`) and iteration limits for the SciPy optimizers.

* **`[extreme_points_data]`**:
    * `hull_points_path`: Path to the JSON file containing known extreme points.
    * `points_to_check_path`: Path to the JSON data file you want to check against the hull.
    * `num_furthest_points`: How many points outside the hull to report.

---

## 🚀 Workflow and Usage

### 1. Running the Optimizer

The main experimental workflow is managed by `optimizer_main.py`.

a.  **Configure `config.toml`** with your desired matrix size, type, and optimization parameters.
b.  **Execute the script** from your terminal:
    ```bash
    python optimizer_main.py
    ```
c.  **Output**: The script will:
    * Log detailed progress to the console and a time-stamped log file in the `logs/` directory.
    * Save the results in a structured JSON file within a directory corresponding to the matrix type (e.g., `sniep/data/`). Each entry in the JSON contains the resulting coefficients, the matrix variables, and the final eigenvalues.
    * If `plot_with_optimize` is `true`, it will generate and save scatter plots of the solution space to a corresponding `plots/` directory.

### 2. Analyzing Extreme Points

Once you have generated data, you can use `extreme_points.py` to analyze its geometric properties.

a.  **Configure `config.toml`**, specifically the `[extreme_points_data]` section, to point to your set of known vertices and the dataset you want to analyze.
b.  **Execute the script**:
    ```bash
    python extreme_points.py
    ```
c.  **Output**: The script prints an analysis to the console, including:
    * A list of necessary vs. redundant points.
    * The top points from your dataset that lie furthest outside the current hull.
    * The equations defining the exposed facets of the hull.
    * It also saves a new file, `extreme_points/points_outside_hull.json`, containing all points found outside the current hull.

### 3. Plotting Existing Data

If you have already generated a JSON data file and want to re-run the plotting stage, you can use the `plot_data.py` script.

a.  **Configure `config.toml`**, setting the `[plot_data]` `data_location` to the path of your JSON file.
b.  **Execute the script**:
    ```bash
    python plot_data.py
    ```

All plotting is done using the plotly library, this gives html files for ease of manipulating the high dimensional data.

---

## 💡 Technical Implementation Details

* **Performance**: To achieve high performance, the computationally expensive functions for the principal minors ($S_k$), their Jacobians, and their Hessians have been symbolically pre-calculated and generated as Python code. These are then Just-In-Time (JIT) compiled using `numba` for near-native C speed. This avoids the massive overhead of symbolic computation during the optimization loops. Optimization is first attempted using `SLSQP`, then if that fails it moves to the slower `trust-constr` method. 
* **Modularity**: The logic is separated into a `lib/` directory, with distinct tasks for file I/O, optimization, eigenvalue computation, and plotting, making the codebase clean and maintainable.
* **Logging**: With each run a comprehensive log file is built. This gives exact details as to potential numerical or other problems that may arise in higher dimension runs.
