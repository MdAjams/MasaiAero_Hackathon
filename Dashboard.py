import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

df = pd.read_csv("Aviation_KPIs_Dataset.csv")

# Convert datetime 
df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"], errors='coerce', format="%m/%d/%Y")
df["Scheduled Month"] = df["Scheduled Departure Time"].dt.month

#Numeric columns for correlation
df_numeric = df.select_dtypes(include=['number'])

# Create Dash app
app = dash.Dash(__name__)


# Dashboard Layout
app.layout = html.Div([
    html.H1("Airline Profitability Dashboard", style={'textAlign': 'center'}),

    # Profit Trend Line Chart
    dcc.Graph(id='profit-trend', figure=px.line(df, x="Scheduled Departure Time", y="Profit (USD)", 
                                                title="Profit Trend Over Time")),

    # Correlation Heatmap (Ensuring numerical data is used)
    dcc.Graph(id='correlation-heatmap', figure=px.imshow(df_numeric.corr(), 
                                                         title="Feature Correlation Heatmap", 
                                                         color_continuous_scale='bluered')),

    # Revenue & Operating Cost Distribution
    dcc.Graph(id='revenue-distribution', figure=px.histogram(df, x="Revenue (USD)", title="Revenue Distribution")),
    dcc.Graph(id='cost-distribution', figure=px.histogram(df, x="Operating Cost (USD)", title="Operating Cost Distribution")),
])

# Run Dashboard
if __name__ == '__main__':
    print("🚀 Dash App is running at: http://127.0.0.1:8050/")
    app.run_server(debug=True)
