import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load dataset
df = pd.read_csv("Aviation_KPIs_Dataset.csv")

# Convert datetime columns
df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"])
df["Scheduled Month"] = df["Scheduled Departure Time"].dt.month

# Function to generate PDF report
def generate_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Airline Profitability Report", ln=True, align='C')
    pdf.ln(10)

    # Summary Statistics
    summary = df[["Profit (USD)", "Revenue (USD)", "Operating Cost (USD)"]].describe()
    
    pdf.cell(200, 10, "Summary Statistics:", ln=True)
    pdf.ln(5)
    
    for col in summary.columns:
        pdf.cell(200, 10, f"{col} - Mean: {summary[col]['mean']:.2f}, Std: {summary[col]['std']:.2f}", ln=True)

    # Save PDF
    pdf.output("Airline_Profitability_Report.pdf")

# Generate Report
generate_report()
print("✅ Report saved as 'Airline_Profitability_Report.pdf'")
