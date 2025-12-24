import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def run_evaluation():
    print("[▶] Running simplified evaluation...")

    # ----------------------------------------
    # 1. Load model output
    # ----------------------------------------
    df = pd.read_csv("output/candidates.csv")

    if df.empty:
        raise ValueError("No contradiction candidates found.")

    # Data hygiene (NOT filtering by label)
    df = df.dropna(
        subset=["sentence_A_text", "sentence_B_text", "contradiction"]
    )

    # ----------------------------------------
    # 2. Load embedding model
    # ----------------------------------------
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ----------------------------------------
    # 3. Compute semantic similarity
    # ----------------------------------------
    emb_a = model.encode(
        df["sentence_A_text"].tolist(),
        show_progress_bar=True
    )
    emb_b = model.encode(
        df["sentence_B_text"].tolist(),
        show_progress_bar=True
    )

    df["semantic_similarity"] = [
        float(cosine_similarity([a], [b])[0][0])
        for a, b in zip(emb_a, emb_b)
    ]

    # ----------------------------------------
    # 4. Evaluation thresholds (operating point)
    # ----------------------------------------
    SIM_THRESHOLD = 0.7
    CONTRA_THRESHOLD = 0.9

    # ----------------------------------------
    # 5. Automatic Evidence Hit Rate (Auto-EHR)
    # ----------------------------------------
    df["valid_evidence"] = (
        (df["semantic_similarity"] >= SIM_THRESHOLD) &
        (df["contradiction"] >= CONTRA_THRESHOLD)
    )

    auto_ehr = df["valid_evidence"].mean()

    # ----------------------------------------
    # 6. High-confidence contradiction rate
    # ----------------------------------------
    high_confidence_rate = (
        df["contradiction"] >= CONTRA_THRESHOLD
    ).mean()

    # ----------------------------------------
    # 7. Average semantic similarity
    # ----------------------------------------
    avg_semantic_similarity = df["semantic_similarity"].mean()

    # ----------------------------------------
    # 8. Final evaluation metrics
    # ----------------------------------------
    results = {
        "Auto_EHR": round(auto_ehr, 4),
        "Avg_Semantic_Similarity": round(avg_semantic_similarity, 4),
        "High_Confidence_Contradictions": round(high_confidence_rate, 4),
        "Similarity_Threshold": SIM_THRESHOLD,
        "Contradiction_Threshold": CONTRA_THRESHOLD,
    }

    # Save results
    pd.DataFrame([results]).to_csv(
        "output/evaluation_metrics.csv",
        index=False
    )

    print("[✓] Evaluation completed.")
    print(results)


if __name__ == "__main__":
    run_evaluation()
