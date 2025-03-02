import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("Aviation_KPIs_Cleaned.csv")

features = ['Delay (Minutes)', 'Aircraft Utilization (Hours/Day)', 'Turnaround Time (Minutes)',
            'Load Factor (%)', 'Fleet Availability (%)', 'Maintenance Downtime (Hours)',
            'Fuel Efficiency (ASK)', 'Revenue (USD)', 'Operating Cost (USD)',
            'Net Profit Margin (%)', 'Ancillary Revenue (USD)', 'Debt-to-Equity Ratio',
            'Revenue per ASK', 'Cost per ASK']
target = 'Profit (USD)'
# Split dataru
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

linear_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = random_forest_model.predict(X_test_scaled)

# Model evaluation
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Streamlit UI
st.title("Aviation KPI Dashboard")
st.sidebar.header("Model Performance")
st.sidebar.write("### Linear Regression")
st.sidebar.write(f"Mean Absolute Error: {mae_linear:.2f}")
st.sidebar.write(f"R-squared: {r2_linear:.2f}")
st.sidebar.write("### Random Forest")
st.sidebar.write(f"Mean Absolute Error: {mae_rf:.2f}")
st.sidebar.write(f"R-squared: {r2_rf:.2f}")

# Visualization
st.subheader("Feature Impact on Profit")
feature_importance = pd.Series(random_forest_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots()
feature_importance.plot(kind='barh', ax=ax, color='royalblue')
st.pyplot(fig)

# Additional visualizations
st.subheader("Data Visualizations")

# Histogram of Profit
graph1 = plt.figure()
sns.histplot(df['Profit (USD)'], bins=30, kde=True, color='blue')
st.pyplot(graph1)

# Correlation Heatmap
graph2 = plt.figure(figsize=(10,6))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(graph2)

# Scatter plot: Revenue vs. Profit
graph3 = plt.figure()
sns.scatterplot(x=df['Revenue (USD)'], y=df['Profit (USD)'], alpha=0.5)
st.pyplot(graph3)

# Boxplot for Delay Minutes
graph4 = plt.figure()
sns.boxplot(x=df['Delay (Minutes)'], color='orange')
st.pyplot(graph4)

# Line plot: Load Factor vs. Profit
graph5 = plt.figure()
sns.lineplot(x=df['Load Factor (%)'], y=df['Profit (USD)'], marker='o')
st.pyplot(graph5)

# Pairplot of selected features
selected_features = ['Revenue (USD)', 'Operating Cost (USD)', 'Profit (USD)', 'Load Factor (%)']
st.subheader("Pairplot of Key Features")
graph6 = sns.pairplot(df[selected_features])
st.pyplot(graph6)

# User input for prediction
st.subheader("Predict Profit")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

# Convert input to DataFrame
user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)
predicted_profit_linear = linear_model.predict(user_df_scaled)[0]
predicted_profit_rf = random_forest_model.predict(user_df_scaled)[0]

st.sidebar.subheader("Predicted Profit")
st.sidebar.write(f"### Linear Regression: ${predicted_profit_linear:,.2f}")
st.sidebar.write(f"### Random Forest: ${predicted_profit_rf:,.2f}")

######################################################################################################################################################
# Checking for Multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Features"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values in Streamlit
st.subheader("Variance Inflation Factor (VIF)")
st.write(vif_data)
########################################################################################################
#Checking for Overfitting
# Model evaluation on training data
y_train_pred_linear = linear_model.predict(X_train_scaled)
y_train_pred_rf = random_forest_model.predict(X_train_scaled)

mae_train_linear = mean_absolute_error(y_train, y_train_pred_linear)
r2_train_linear = r2_score(y_train, y_train_pred_linear)

mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)

# Display Train vs Test performance
st.sidebar.subheader("Overfitting Check")
st.sidebar.write("### Linear Regression")
st.sidebar.write(f"Train R²: {r2_train_linear:.2f}, Test R²: {r2_linear:.2f}")
st.sidebar.write("### Random Forest")
st.sidebar.write(f"Train R²: {r2_train_rf:.2f}, Test R²: {r2_rf:.2f}")
############################################################################################################################

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to remove high-VIF features
def remove_high_vif_features(X, threshold=5.0):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    while vif_data["VIF"].max() > threshold:
        # Remove the feature with the highest VIF
        highest_vif_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"]
        X = X.drop(columns=[highest_vif_feature])
        
        # Recompute VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return X

# Remove high-VIF features
X_filtered = remove_high_vif_features(X)

# Updated feature set
st.subheader("Selected Features After Removing High VIF")
st.write(X_filtered.columns.tolist())

# Update X_train, X_test
X_train_filtered, X_test_filtered, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# Train models with selected features
linear_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)
