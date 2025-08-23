from flask import Flask, request, jsonify
from flask_cors import CORS
import io, csv
import pandas as pd
from .nlp_refiner import predict_descriptions, learn_feedback, labels as nlp_labels
import numpy as np
from .savings import get_savings_suggestions
# server/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

# IMPORTANT: relative imports because we're running as package "server"
from .nlp_refiner import predict_descriptions, learn_feedback, labels as nlp_labels
from .savings import get_savings_suggestions


# Import your ML wrapper that exposes predict_categories(df) -> pd.Series[str]
from .machinelearningclassification import predict_categories

app = Flask(__name__)
CORS(app)  # allow localhost frontends (Electron/React)

def summarize_by_category(df: pd.DataFrame, cat_col: str):
    # Ensure numeric amounts
    df["Amount"] = pd.to_numeric(df.get("Amount", 0), errors="coerce").fillna(0.0)

    groups = df.groupby(cat_col)["Amount"]
    out = []
    for cat, series in groups:
        total = float(series.sum())
        deposits = float(series[series >= 0].sum())
        withdrawals = float(series[series < 0].sum())
        out.append({
            "Category": str(cat),
            "TransactionCount": int(series.shape[0]),
            "TotalAmount": total,
            "Withdrawals": withdrawals,
            "Deposits": deposits,
        })
    return out

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/upload-csv")
def upload_csv():
    """
    Accepts a CSV with columns like: Date, Description, Amount, Category
    Runs ML to predict categories and returns:
      - category_summary (based on ML predictions)
      - entries_with_pred (rows + PredictedCategory)
    """
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400

    file = request.files["file"]
    content = file.read().decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    if not rows:
        return jsonify({"category_summary": [], "entries_with_pred": []})

    # Detect header (very lightweight check)
    header = rows[0]
    lower = [str(h or "").strip().lower() for h in header]
    looks_like_header = len(lower) >= 3 and ("date" in lower[0] and "amount" in lower[2])

    data_rows = rows[1:] if looks_like_header else rows
    columns = header if looks_like_header else ["Date", "Description", "Amount", "Category"]

    df = pd.DataFrame(data_rows, columns=columns).fillna("")

    # Run your ML predictions
    preds = predict_categories(df)  # must be length == len(df)
    df["PredictedCategory"] = preds.astype(str).fillna("Uncategorized")

    # Build ML-based summary
    category_summary = summarize_by_category(df, "PredictedCategory")

    # Return also per-row predictions (handy for a table)
    entries_with_pred = df[["Date", "Description", "Amount", "PredictedCategory"]].to_dict(orient="records")

    return jsonify({
        "category_summary": category_summary,
        "entries_with_pred": entries_with_pred,
    })

@app.post("/nlp/refine")
def nlp_refine():
    """
    Body: { rows: [{Description, Amount}], threshold?: 0.45 }
    Returns: [{ PredictedCategory, Confidence }]
    """
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows", [])
    threshold = float(payload.get("threshold", 0.45))

    df = pd.DataFrame(rows).fillna("")
    out = predict_descriptions(df)  # PredictedCategory + Confidence
    # if below threshold -> keep as 'Uncategorized'
    below = out["Confidence"] < threshold
    out.loc[below, "PredictedCategory"] = "Uncategorized"

    return jsonify(out.to_dict(orient="records"))

@app.post("/nlp/feedback")
def nlp_feedback():
    """
    Body: { samples: [{Description, Amount, CorrectCategory}] }
    Updates the online model (partial_fit) and persists artifacts.
    """
    payload = request.get_json(silent=True) or {}
    samples = pd.DataFrame(payload.get("samples", [])).fillna("")
    if samples.empty:
        return jsonify({"updated": 0})
    learn_feedback(samples)
    return jsonify({"updated": int(samples.shape[0])})

@app.get("/nlp/labels")
def nlp_get_labels():
    return jsonify({"labels": nlp_labels()})

@app.post("/savings/suggestions")
def savings_suggestions():
    try:
        payload = request.get_json(silent=True) or {}
        categories = payload.get("categories", []) or []
        merchants = payload.get("merchants", []) or []
        max_items = int(payload.get("max_items", 12))
        items = get_savings_suggestions(categories, merchants, max_items=max_items)
        return jsonify({"items": items})
    except Exception as e:
        print("savings error:", e)
        return jsonify({"items": []})

if __name__ == "__main__":
    # Run as a script (useful during development)
    app.run(host="127.0.0.1", port=5050, debug=False)
