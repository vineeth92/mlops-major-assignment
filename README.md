MLOps Major Assignment

Submitted By
Name: Vineeth Varghese  
Roll No: G24AI1104  
Email: g24ai1104@iitj.ac.in  


MLOps Linear Regression Pipeline

This repository contains a complete MLOps pipeline for a Linear Regression model, developed as part of a major assignment. The pipeline automates model training, testing, manual quantization, Dockerization, and continuous integration/continuous deployment (CI/CD) using GitHub Actions. All processes are managed within a single main branch.


Project Objective

To build an MLOps pipeline for a Linear Regression model using the California Housing dataset, demonstrating key MLOps practices like:

- Automated testing
- Model serialization
- Manual quantization
- Containerization (Docker)
- CI/CD integration via GitHub Actions

Dataset and Model

Dataset: California Housing dataset (`sklearn.datasets`)
Model: `sklearn.linear_model.LinearRegression`

Project Structure

├── .github/ # GitHub Actions workflows
│ └── workflows/
│ └── ci.yml # CI/CD pipeline definition
├── src/ # Source code for the ML pipeline
│ ├── init.py # Marks 'src' as a Python package
│ ├── train.py # Script for model training
│ ├── quantize.py # Script for manual model quantization
│ └── predict.py # Script for model inference (used in Docker)
├── tests/ # Unit tests for the pipeline
│ ├── init.py # Marks 'tests' as a Python package
│ └── test_train.py # Tests for model training and data loading
├── Dockerfile # Defines the Docker image for inference
├── requirements.txt # Python dependencies
├── .gitignore # Files/directories to ignore in Git
└── README.md # Project description and comparison table


CI/CD Pipeline Overview

The CI/CD pipeline is defined in `.github/workflows/ci.yml` and consists of three sequential jobs:

| Job Name               | Description                                                                               | Depends On         |
|------------------------|-------------------------------------------------------------------------------------------|--------------------|
| `test_suite`           | Runs `pytest` to ensure code quality and functionality. Must pass before others execute. | None               |
| `train_and_quantize`   | Trains the model, performs quantization, and uploads model artifacts.                    | `test_suite`       |
| `build_and_test_container` | Builds Docker image and verifies `predict.py` runs inside the container.             | `train_and_quantize` |


Model Performance and Size Comparison

| Metric/Component         | Original Model (`sklearn_model.joblib`) | Quantized Model (`quant_params.joblib`)     |
|--------------------------|------------------------------------------|---------------------------------------------|
| R² Score (Test Set)      | 0.5758                                   | -1069.3508                                   |
| MSE (Test Set)           | 0.5559                                   | 1402.5973                                    |
| File Size                | 0.68 KB                                  | 0.4 KB                                       |
| Parameters Stored        | Full floating-point `coef_`, `intercept_` | Unsigned 8-bit integers + scale/zero_point  |
| Inference Precision      | Standard floating-point precision        | Reduced precision due to quantization       |
| Primary Use Case         | General-purpose inference, higher precision | Resource-constrained environments (low size) |

