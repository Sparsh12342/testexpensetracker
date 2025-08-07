import csv
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# ---------- Helper Functions ----------

def group_by_first_word(df):
    df['FirstWord'] = df['Description'].str.split().str[0]

    deposits = df[df['Amount'] > 0]
    withdrawals = df[df['Amount'] < 0]

    deposit_summary = deposits.groupby('FirstWord')['Amount'].sum().reset_index()
    deposit_summary = deposit_summary.rename(columns={'Amount': 'TotalDeposits'})

    withdrawal_summary = withdrawals.groupby('FirstWord')['Amount'].sum().reset_index()
    withdrawal_summary = withdrawal_summary.rename(columns={'Amount': 'TotalWithdrawals'})

    deposit_dict = deposit_summary.to_dict(orient='records')
    withdrawal_dict = withdrawal_summary.to_dict(orient='records')

    return deposit_dict, withdrawal_dict


def group_by_cluster(df):
    if "Subcluster_Label" in df.columns:
        cluster_col = "Subcluster_Label"
    elif "BERT_Category" in df.columns:
        cluster_col = "BERT_Category"
    else:
        return [], []

    deposits = df[df["Amount"] > 0]
    withdrawals = df[df["Amount"] < 0]

    deposit_summary = deposits.groupby(cluster_col)["Amount"].sum().reset_index()
    deposit_summary = deposit_summary.rename(columns={"Amount": "TotalDeposits", cluster_col: "Cluster_Label"})

    withdrawal_summary = withdrawals.groupby(cluster_col)["Amount"].sum().reset_index()
    withdrawal_summary = withdrawal_summary.rename(columns={"Amount": "TotalWithdrawals", cluster_col: "Cluster_Label"})

    deposit_dict = deposit_summary.to_dict(orient="records")
    withdrawal_dict = withdrawal_summary.to_dict(orient="records")

    return deposit_dict, withdrawal_dict


def process_dataframe(df):
    df = df[df["Amount"].astype(str).str.strip() != ""]
    df["Amount"] = df["Amount"].astype(str).str.replace(",", "")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    deposits = df[df["Amount"] > 0]
    withdrawals = df[df["Amount"] < 0]

    total_deposits = round(deposits["Amount"].sum(), 2)
    total_withdrawals = round(withdrawals["Amount"].sum(), 2)

    num_deposits = len(deposits)
    num_withdrawals = len(withdrawals)

    deposit_firstword, withdrawal_firstword = group_by_first_word(df)
    deposit_clusters, withdrawal_clusters = group_by_cluster(df)

    category_summary = []
    if "BERT_Category" in df.columns:
        for category, group in df.groupby("BERT_Category"):
            count = group.shape[0]
            total = group["Amount"].sum()
            withdrawals_cat = group[group["Amount"] < 0]["Amount"].sum()
            deposits_cat = group[group["Amount"] >= 0]["Amount"].sum()

            category_summary.append({
                "Category": category,
                "TransactionCount": count,
                "TotalAmount": total,
                "Withdrawals": withdrawals_cat,
                "Deposits": deposits_cat
            })

    return {
        "total_deposits": total_deposits,
        "total_withdrawals": total_withdrawals,
        "num_deposits": num_deposits,
        "num_withdrawals": num_withdrawals,
        "deposits_grouped_by_first_word": deposit_firstword,
        "withdrawals_grouped_by_first_word": withdrawal_firstword,
        "deposits_grouped_by_cluster": deposit_clusters,
        "withdrawals_grouped_by_cluster": withdrawal_clusters,
        "category_summary": category_summary
    }


# ---------- Endpoints ----------
@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save uploaded file
        upload_path = "uploaded.csv"
        file.save(upload_path)

        # Process it
        result = subprocess.run(
            ["python3", "machinelearningclassification.py", upload_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return jsonify({"error": f"ML script failed:\n{result.stderr}"}), 500

        # Read labeled output and send processed summary
        df = pd.read_csv("stmt_clustered_labeled.csv")
        return jsonify(process_dataframe(df))

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 400



@app.route('/')
def home():
    return "✅ Flask backend is running."


# ---------- App Runner ----------

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://localhost:5050")
    app.run(debug=True, port=5050)


# import csv
# import pandas as pd
# from flask import Flask, jsonify

# app = Flask(__name__)

# def group_by_first_word(df):
#     # Extract first word of description into a new column
#     df['FirstWord'] = df['Description'].str.split().str[0]

#     # Separate deposits and withdrawals
#     deposits = df[df['Amount'] > 0]
#     withdrawals = df[df['Amount'] < 0]

#     # Group deposits by first word and sum amounts
#     deposit_summary = deposits.groupby('FirstWord')['Amount'].sum().reset_index()
#     deposit_summary = deposit_summary.rename(columns={'Amount': 'TotalDeposits'})

#     # Group withdrawals by first word and sum amounts
#     withdrawal_summary = withdrawals.groupby('FirstWord')['Amount'].sum().reset_index()
#     withdrawal_summary = withdrawal_summary.rename(columns={'Amount': 'TotalWithdrawals'})

#     # Convert to dictionary format for JSON
#     deposit_dict = deposit_summary.to_dict(orient='records')
#     withdrawal_dict = withdrawal_summary.to_dict(orient='records')

#     return deposit_dict, withdrawal_dict

# @app.route('/summary')
# def summary():
#     filename = "stmt.csv"
#     cleaned_data = []

#     # Step 1: Read valid rows only
#     with open(filename, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if len(row) == 4:
#                 cleaned_data.append(row)

#     # Step 2: Build DataFrame
#     df = pd.DataFrame(cleaned_data[1:], columns=["Date", "Description", "Amount", "Running Bal."])

#     # Step 3: Clean and convert Amount column
#     df = df[df["Amount"].str.strip() != ""]  # remove empty string amounts
#     df["Amount"] = df["Amount"].str.replace(",", "")  # remove commas
#     df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")  # convert to float safely
#     df = df.dropna(subset=["Amount"])  # drop rows where conversion failed

#     # Step 4: Basic totals and counts
#     deposits = df[df["Amount"] > 0]
#     withdrawals = df[df["Amount"] < 0]

#     total_deposits = round(deposits["Amount"].sum(), 2)
#     total_withdrawals = round(withdrawals["Amount"].sum(), 2)

#     # Step 5: Group by first word summaries
#     deposit_groups, withdrawal_groups = group_by_first_word(df)

#     summary = {
#         "total_deposits": total_deposits,
#         "total_withdrawals": total_withdrawals,
#         "num_deposits": len(deposits),
#         "num_withdrawals": len(withdrawals),
#         "deposits_grouped_by_first_word": deposit_groups,
#         "withdrawals_grouped_by_first_word": withdrawal_groups
#     }

#     return jsonify(summary)

# if __name__ == "__main__":
#     app.run(debug=True, port=5050)









# from flask import Flask, jsonify
# from flask_cors import CORS
# import pandas as pd

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:5173"])  

# @app.route('/summary', methods=['GET'])
# def summary():   
#     filename = "stmt.csv"  

#     try:
#         df = pd.read_csv(filename)
#         df = df[pd.to_numeric(df["Amount"].str.replace(",", ""), errors="coerce").notna()]
#         df["Amount"] = df["Amount"].str.replace(",", "").astype(float)
#         df = df[["Date", "Description", "Amount"]]  # optional: drop "Running Bal."
#         return df.to_json(orient="records")
#     except Exception as e:
#         return {"error": str(e)}, 500
#     except FileNotFoundError:
#         return jsonify(["CSV file not found."])

# #removing all of the commas
#     df['Amount'] = df["Amount"].replace(',', '', regex=True)
#     df['RunningBal.'] = pd.to_numeric(df['RunningBal.'], errors='coerce')
#     df['RunningBal.'] = df["RunningBal."].replace(',', '', regex=True)
#     df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

#     output = []


# #grouping da deposits with da descriptions
#     output.append("Total deposits per description:")
#     deposits = df.groupby('Description')['Amount'].sum()
#     for desc, amount in amount.items():
#         output.append(f"{desc}: ${float(amount):.2f}")

#     output.append("Total withdrawals per description:")
#     withdrawals = df.groupby('Description')['Withdrawls'].sum()
#     for desc, amount in withdrawals.items():
#         output.append(f"{desc}: ${float(amount):.2f}")

#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True, port=5050)














# def get_expense_summary(transactions):
#     deposit_count = 0
#     withdrawal_count = 0

#     for t in transactions:
#         description = t['description'].lower()
#         amount = t['amount']
#         if "deposit" in description:
#             deposit_count += amount
#         elif "withdrawal" in description:
#             withdrawal_count += amount

#     summary = f"Total Deposits: ${deposit_count:.2f}\nTotal Withdrawals: ${withdrawal_count:.2f}"
#     return summary




##def main():
#        tracker= ExpenseTracker()
#
#        while True:
 #           print("\nExpense Tracker Menu")
 #           print("1. add expense")
  #          print("2. remove expense")
   #         print("3. view expenses")
 #           print("4. total expenses ")
#            print("leave")

 #           choice = input("Enter your choice (1-5):")
 #           if choice == "1":
 #               date = input("Enter the date (YYYY-MM-DD)")
 #               description = input("Enter the description: ")
 #               amount = float(input("Enter the amount: "))
 #               expense = Expense(date, description, amount)
 #               tracker.add_expense(expense)
 #               print("expense added successfully")
 #           elif choice == "2":
 #               index = int(input(" enter the index number to remove: ")) -1 
 #               tracker.remove_expense(index)
 #           elif choice == "3":
 #               tracker.view_expenses()
  #          elif choice == "4":
 #               tracker.total_expenses()
 #           elif choice == "5":
 #               print( "ok lit")
 #               break 
 #           else:
 #               print ("idk what u are trying to say")

