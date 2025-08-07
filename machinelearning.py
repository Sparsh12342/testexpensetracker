import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

FILENAME = "stmt.csv"
MAX_CLUSTERS = 10

def get_top_keywords_per_cluster(tfidf_matrix, labels, vectorizer, top_n=3):
    df = pd.DataFrame(tfidf_matrix.todense()).groupby(labels).mean()
    terms = vectorizer.get_feature_names_out()
    top_keywords = {}
    for i, row in df.iterrows():
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [terms[idx] for idx in top_indices]
        top_keywords[i] = ", ".join(keywords)
    return top_keywords

def generate_cluster_label(keywords):
    if not keywords:
        return "Miscellaneous"

    if "uber" in keywords or "lyft" in keywords:
        return "Rideshare and travel spending"
    if "amazon" in keywords or "order" in keywords:
        return "Online shopping expenses"
    if "walmart" in keywords or "target" in keywords:
        return "Retail store purchases"
    if "atm" in keywords or "withdrawal" in keywords:
        return "Cash withdrawals"
    if "deposit" in keywords or "payroll" in keywords:
        return "Income or direct deposits"
    if "starbucks" in keywords or "coffee" in keywords:
        return "Café or food purchases"

    return f"Transactions related to: {', '.join(keywords)}"

def cluster_descriptions():
    df = pd.read_csv(FILENAME, on_bad_lines='skip')
    descriptions = df["Description"].fillna("").tolist()

    if len(descriptions) < 2:
        print("Not enough descriptions to cluster.")
        return

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(descriptions)

    # Determine best number of clusters
    best_k = 2
    best_score = -1
    for k in range(2, min(MAX_CLUSTERS, len(descriptions)) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score

    print(f"\nBest number of clusters: {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    # Label each cluster with keywords
    top_keywords = get_top_keywords_per_cluster(X, df["Cluster"], vectorizer)

    # Add readable labels using NLP-style description
    df["Cluster_Label"] = df["Cluster"].apply(
        lambda x: generate_cluster_label(top_keywords[x].split(", "))
    )

    # Print labeled samples
    for cluster_num in range(best_k):
        keywords = top_keywords[cluster_num].split(", ")
        label = generate_cluster_label(keywords)
        print(f"\nCluster {cluster_num} ({label}) samples:")
        samples = df[df["Cluster"] == cluster_num]["Description"].head(5).tolist()
        for desc in samples:
            print(f" - {desc}")

    # Save to CSV
    df.to_csv("stmt_clustered_labeled.csv", index=False)
    print("\nClustered and labeled data saved to stmt_clustered_labeled.csv")

if __name__ == "__main__":
    cluster_descriptions()
