# README: Airline Profitability Prediction Model

## Project Overview
This project aims to develop a machine learning model to predict airline profitability based on historical flight performance and operational data. The model helps airline operators understand the key drivers of profitability, optimize operations, and make data-driven business decisions.

## Objective
The primary objective is to create a **high-performance ML model** that accurately predicts **Profit (USD)** for each flight while providing actionable insights to improve operational efficiency and profitability.

## Dataset Description
The dataset includes the following key features:
- Flight delays (Minutes)
- Aircraft utilization (Hours/Day)
- Turnaround time (Minutes)
- Load factor (%)
- Fleet availability (%)
- Maintenance downtime (Hours)
- Fuel efficiency (ASK)
- Revenue (USD)
- Operating costs (USD)
- Net profit margin (%)
- Ancillary revenue (USD)
- Debt-to-equity ratio
- Revenue per ASK
- Cost per ASK
- Profit (USD) (Target Variable)
## Deplolyed App
** Machine learning Model 

## Methodology
### 1. Data Preprocessing
- Handling missing values
- Outlier detection and treatment
- Feature scaling (Standardization)
- Time-based feature engineering

### 2. Exploratory Data Analysis (EDA)
- Correlation analysis
- Visualization of feature distributions
- Seasonal pattern identification

### 3. Feature Selection
- Statistical tests
- Recursive Feature Elimination (RFE)
- Feature importance from tree-based models

### 4. Model Development
Models considered:
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks

### 5. Model Evaluation
Metrics used:
- R-Squared
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### 6. Explainability
- SHAP values
- Feature importance plots
- Partial dependence plots
- LIME explanations

### 7. Deployment
- API development for real-time predictions
- Visualization dashboard
- Automated model retraining pipeline

## How to Run the Project
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the model training script:
```bash
python train_model.py
```
3. Test the model:
```bash
python test_model.py
```
4. Launch the API service:
```bash
python app.py
```

## Results
The model will generate predictions and provide feature importance insights to help operators optimize operational efficiency.

## Folder Structure
```
├── data                # Raw and processed data
├── notebooks           # Jupyter notebooks for EDA and experimentation
├── models              # Trained models
├── scripts             # Python scripts for training and evaluation
└── README.md           # Project documentation
```

## Future Enhancements
- Dynamic pricing optimization
- Demand forecasting
- Real-time anomaly detection

## Contributors
- Jasveer Singh
- Akash Singh Rathour

## License
This project is licensed under the MIT License.
