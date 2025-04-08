# GSoC 2025 Proof-of-Concept: Aeon Project #2 - ML Forecasting Evaluation

**Applicant:** Somto Onyekwelu ([https://github.com/SomtoOnyekwelu](https://github.com/SomtoOnyekwelu))
**GSoC Organization:** NumFOCUS (aeon project)
**Target Project Idea:** Project #2: Forecasting - Implementing and evaluating machine learning forecasters

---

## Overview

This repository contains **Proof-of-Concept (PoC) code** developed in support of my Google Summer of Code 2025 application to the **aeon** project under NumFOCUS. It specifically demonstrates foundational capabilities relevant to **Project #2: Forecasting - Implementing and evaluating machine learning forecasters**.

The full GSoC project proposal aims to:
1.  Implement the **SETAR-Tree** forecasting algorithm within `aeon`.
2.  Rigorously **evaluate** SETAR-Tree and selected existing `aeon` regressors (used as forecasters) on benchmark datasets.
3.  Uniquely analyze the **robustness** of these forecasters to noisy input time series data.
4.  Contribute necessary **framework improvements** (e.g., reusable feature engineering utilities).
5.  *(Stretch Goal):* Implement a baseline **LightGBM forecaster wrapper**.

This PoC repository **focuses on demonstrating core skills** required for these tasks, particularly time series data handling, basic feature engineering (lagging), applying an ML model in a forecasting context, and performing simple evaluations using Python, Pandas, and Scikit-learn.

## Purpose of this PoC Repository

*   **Demonstrate Capability:** Show practical Python skills for time series manipulation and basic ML workflows.
*   **Illustrate Understanding:** Provide concrete examples of foundational concepts (lagging, train/test splits for time series) required for the full GSoC project.
*   **Show Proactivity:** Serve as tangible evidence of preparation and commitment specifically tailored to `aeon`'s Project #2 requirements.
*   **Support GSoC Proposal:** Act as a direct, verifiable reference linked within the formal GSoC proposal PDF.

*Note: This code is intentionally simplified for demonstration. The full GSoC project involves deeper integration with `aeon`'s API, robust time series cross-validation, implementation of target algorithms like SETAR-Tree, advanced feature engineering, and comprehensive robustness analysis.*

## Code Structure

*   **`src/forecasting_poc.py`**: The main script demonstrating the workflow. Takes command-line arguments for flexibility.
    *   Loads data from a CSV file (default: `data/sample_air_quality.csv`).
    *   Performs basic cleaning and prepares a time series.
    *   Generates lagged features from the time series.
    *   Splits data chronologically for training and testing.
    *   Trains a placeholder `RandomForestRegressor` model.
    *   Predicts on the test set and evaluates using MAE & RMSE.
*   **`data/sample_air_quality.csv`**: Sample hourly NO2 data (from OpenAQ Paris data) used by the script's default settings.
*   **`requirements.txt`**: Lists necessary Python packages (`pandas`, `numpy`, `scikit-learn`).

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SomtoOnyekwelu/gsoc-2025-aeon-ml-forecast-evaluation.git
    cd gsoc-2025-aeon-ml-forecast-evaluation
    ```
2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the PoC script (using defaults):**
    ```bash
    python src/forecasting_poc.py
    ```
5.  **Run with custom parameters (Example):**
    ```bash
    python src/forecasting_poc.py --data path/to/my_data.csv --time_col timestamp --target_col measurement --n_lags 10 --filter_col sensor_id --filter_val sensor_ABC
    ```
    *Use `python src/forecasting_poc.py --help` to see all options.*

---

Thank you for reviewing this supporting code. I am very excited about the prospect of contributing to the `aeon` project through GSoC 2025!