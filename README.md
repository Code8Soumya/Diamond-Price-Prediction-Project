# Diamond Price Prediction

Welcome to the Diamond Price Prediction project! This project leverages regression techniques to predict the price of diamonds. Below is an overview of the key features, and usage instructions.

## Key Features

1. **Modular Approach**: The project is designed with a modular structure for ease of maintenance and scalability.
2. **Exploratory Data Analysis (EDA)**: Utilizes `pandas`, `numpy`, and `matplotlib` for data visualization and EDA.
3. **Modeling**: Uses the `xgboost` model for accurate price predictions.
4. **Categorical Feature Handling**: Employs label encoding for categorical features.
5. **Pipeline Creation**: Includes separate pipelines for data ingestion, transformation, and training/testing.
6. **Exception Handling and Logging**: Implements robust exception handling and logging mechanisms.
7. **Flask Frontend**: Provides a user-friendly Flask web application for predictions.
8. **Package Structure**: The entire project is structured as a Python package.

## Installation and Usage

Follow these steps to set up and run the Diamond Price Prediction project.

### Step 1: Clone the project
```bash
git clone https://github.com/Code8Soumya/Diamond-Price-Prediction-Project.git
```

### Step 2: Creating environment and installing required packages
```bash
conda create -p venv python==3.12 -y
```
```bash
pip install -r requirements.txt
```

### Step 3: Training the model
```bash
python src/pipelines/training_pipeline.py
```

### Step 4: Running the model for predictions
```bash
python application.py
```

Go to https://localhost:5000/predict for prediction




