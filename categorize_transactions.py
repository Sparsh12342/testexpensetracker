# categorize_transactions.py

import pandas as pd
import joblib

# Load model
model = joblib.load("transaction_classifier.joblib")

# Load bank statement
stmt_df = pd.read_csv("stmt.csv")
stmt_df["Description"] = stmt_df["Description"].fillna("")

# Predict categories
stmt_df["Predicted_Category"] = model.predict(stmt_df["Description"])

# Optional: sort by category and total
if "Amount" in stmt_df.columns:
    summary = stmt_df.groupby("Predicted_Category")["Amount"].sum().sort_values(ascending=False)
    print("\n💸 Total Spending by Category:\n")
    print(summary)

# Save labeled data
stmt_df.to_csv("stmt_with_categories.csv", index=False)
print("\n✅ Saved to stmt_with_categories.csv")
