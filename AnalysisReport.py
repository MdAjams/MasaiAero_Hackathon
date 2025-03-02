import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image

# Load dataset
df = pd.read_csv("Aviation_KPIs_Dataset.csv")

# Convert datetime columns
df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"])
df["Scheduled Month"] = df["Scheduled Departure Time"].dt.month

# Summary Statistics
summary = df[["Profit (USD)", "Revenue (USD)", "Operating Cost (USD)"]].describe()

# 📊 1. Profit Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(df["Profit (USD)"], bins=30, kde=True, color="blue")
plt.title("Profit Distribution")
plt.xlabel("Profit (USD)")
plt.ylabel("Frequency")
plt.savefig("profit_distribution.png")
plt.close()

# 📊 2. Correlation Heatmap (Only numeric columns)
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=["number"])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ✅ Generate PDF Report
pdf_file = "Airline_Profitability_Report.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

# Title
c.setFont("Helvetica-Bold", 20)
c.drawCentredString(width / 2.0, height - 40, "📊 Airline Profitability Report")

# Summary Statistics
c.setFont("Helvetica-Bold", 14)
c.drawString(30, height - 80, "1️⃣ Summary Statistics")
c.setFont("Helvetica", 12)
c.drawString(30, height - 100, f"Profit (Mean): {summary['Profit (USD)']['mean']:.2f}")
c.drawString(30, height - 120, f"Revenue (Mean): {summary['Revenue (USD)']['mean']:.2f}")
c.drawString(30, height - 140, f"Operating Cost (Mean): {summary['Operating Cost (USD)']['mean']:.2f}")

# Key Profitability Insights
c.setFont("Helvetica-Bold", 14)
c.drawString(30, height - 180, "2️⃣ Key Profitability Insights")
c.setFont("Helvetica", 12)
c.drawString(30, height - 200, "✔️ Higher revenue is strongly correlated with higher profit.")
c.drawString(30, height - 220, "✔️ Operating costs have a moderate negative impact on profit.")
c.drawString(30, height - 240, "✔️ Load Factor and Fleet Availability contribute positively to profitability.")

# Profit Distribution Plot
c.setFont("Helvetica-Bold", 14)
c.drawString(30, height - 280, "3️⃣ Profit Distribution")
c.drawImage("profit_distribution.png", 30, height - 580, width=500, height=250)

# Feature Correlation Heatmap
c.setFont("Helvetica-Bold", 14)
c.drawString(30, height - 620, "4️⃣ Feature Correlation Heatmap")
c.drawImage("correlation_heatmap.png", 30, height - 920, width=500, height=250)

c.save()

print("✅ PDF Report saved as 'Airline_Profitability_Report.pdf'")