import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
from collections import Counter, defaultdict
import sys

FILENAME = sys.argv[1] if len(sys.argv) > 1 else "stmt.csv"
NUM_SAMPLES = 5
MIN_CLUSTERS = 5
MAX_CLUSTERS = 15

bert_model_dir = "./bert_expense_classifier"  # your path here
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
bert_model.eval()

label_encoder = joblib.load("label_encoder.joblib")  # your path here

def classify_with_bert(descriptions):
    inputs = bert_tokenizer(list(descriptions), padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    categories = label_encoder.inverse_transform(preds)
    return categories

def choose_optimal_clusters(features, min_k=MIN_CLUSTERS, max_k=MAX_CLUSTERS):
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        print(f"Silhouette score for k={k}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Chosen number of clusters: {best_k} with silhouette score {best_score:.4f}")
    return best_k

def most_common_category(categories):
    if len(categories) == 0:
        return "Unknown", 0
    c = Counter(categories)
    cat, count = c.most_common(1)[0]
    return cat, count

def cluster_descriptions():
    df = pd.read_csv(FILENAME, on_bad_lines='skip')
    df["Description"] = df["Description"].fillna("")

    if 'Amount' not in df.columns:
        raise ValueError("CSV must contain an 'Amount' numeric column for spend calculations.")

    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')
    invalid_amounts = df["Amount"].isna().sum()
    if invalid_amounts > 0:
        print(f"Warning: Found {invalid_amounts} invalid amounts, setting them to 0")
        df["Amount"] = df["Amount"].fillna(0)

    if df.shape[0] < 2:
        print("Not enough descriptions to cluster.")
        return

    descriptions = df["Description"].tolist()

    print("🔍 Running BERT classification on descriptions...")
    df["BERT_Category"] = classify_with_bert(descriptions)

    encoder = OneHotEncoder(sparse_output=True)
    bert_cat_encoded = encoder.fit_transform(df[["BERT_Category"]])

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    combined_features = sp.hstack([tfidf_matrix, bert_cat_encoded], format='csr')

    print("\n⚡ Finding optimal number of main clusters...")
    num_main_clusters = choose_optimal_clusters(combined_features)

    print(f"\n⚡ Clustering into {num_main_clusters} main clusters...")
    kmeans = KMeans(n_clusters=num_main_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(combined_features)

    # Main cluster labels and stats
    cluster_labels = {}
    cluster_stats = {}

    for c in sorted(df["Cluster"].unique()):
        cats = df[df["Cluster"] == c]["BERT_Category"]
        cat, cat_count = most_common_category(cats)
        count = len(cats)
        total_spent = df[df["Cluster"] == c]["Amount"].sum()
        cluster_labels[c] = cat
        cluster_stats[c] = {
            "most_common_category": cat,
            "category_count": cat_count,
            "transaction_count": count,
            "total_amount": total_spent
        }

    df["Cluster_Label"] = df["Cluster"].map(cluster_labels)

    print("\nMain Cluster Stats:")
    for c, stats in cluster_stats.items():
        print(f"Cluster {c}:")
        print(f"  Label: {stats['most_common_category']}")
        print(f"  Transactions: {stats['transaction_count']}")
        print(f"  Category Count: {stats['category_count']}")
        print(f"  Total Amount: ${stats['total_amount']:.2f}")

    # Subclusters grouped by first word of description inside each main cluster
    subcluster_labels = {}
    subcluster_stats = {}

    for main_cluster in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == main_cluster].copy()
        # Extract first word (lowercase) from description or 'Unknown'
        cluster_df["First_Word"] = cluster_df["Description"].str.split().str[0].str.lower().fillna("unknown")

        # Group by this first word
        grouped = cluster_df.groupby("First_Word")

        print(f"\nMain Cluster {main_cluster} Subgroups by first word:")

        for first_word, group in grouped:
            cats = group["BERT_Category"]
            cat, cat_count = most_common_category(cats)
            count = len(group)
            total_spent = group["Amount"].sum()
            withdrawals = group[group["Amount"] < 0]["Amount"].sum()
            deposits = group[group["Amount"] >= 0]["Amount"].sum()

            label = cat  # Label subcluster by most common BERT category

            print(f"  Label: {label}")
            print(f"    First word group: '{first_word}'")
            print(f"    Transactions: {count}")
            print(f"    Category Count: {cat_count}")
            print(f"    Total Amount: ${total_spent:.2f}")
            print(f"    Total Withdrawals: ${withdrawals:.2f}")
            print(f"    Total Deposits: ${deposits:.2f}")

            # Save subcluster stats for later if needed
            subcluster_labels[(main_cluster, first_word)] = label
            subcluster_stats[(main_cluster, first_word)] = {
                "most_common_category": label,
                "category_count": cat_count,
                "transaction_count": count,
                "total_amount": total_spent,
                "withdrawals": withdrawals,
                "deposits": deposits,
            }

    # Save detailed output CSV with main cluster and subcluster (first word) labels
    # Map subcluster label per row
    def get_subcluster_label(row):
        return subcluster_labels.get(
            (row["Cluster"], row["Description"].split()[0].lower() if len(row["Description"].split()) > 0 else "unknown"),
            "Unknown"
        )

    df["Subcluster_Label"] = df.apply(get_subcluster_label, axis=1)

    df.to_csv("stmt_clusetered_labeled.csv", index=False)

    print("\n✅ Saved detailed transactions with first-word subclusters to stmt_bert_tfidf_clusters_firstword_subclusters.csv")

if __name__ == "__main__":
    cluster_descriptions()
