import os
from functools import partial
from typing import List

import pandas as pd
from haystack.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

from src import custom_preprocessors, loading, processing, scoring
from src.processing import split_into_sentences, compute_sentence_embeddings
from src.evaluation import run_evaluation
from config import *


def load_documents():
    print("[▶] Loading documents...")
    return loading.load_dataset_from_json(DATASET_FILEPATH)


def preprocess_documents(df):
    print("[▶] Preprocessing documents...")

    # Combine pages
    df["fulltext"] = df.text_by_page.apply(processing.clean_and_combine_pages)

    # Convert to Haystack documents
    docs = processing.convert_frame_to_haystack(df)

    # Sentence-level chunking with overlap
    cleaner = partial(
        processing.clean_sentence_splits,
        toc_period_threshold=CHUNK_CLEANING_TOC_PERIOD_THRESHOLD,
        length_minimum=CHUNK_CLEANING_LENGTH_MINIMUM,
        length_maximum=CHUNK_CLEANING_LENGTH_MAXIMUM,
    )

    chunker = custom_preprocessors.SplitCleanerPreProcessor(
        language="en",
        split_by="sentence",
        split_cleaner=cleaner,
        split_length=CHUNK_LENGTH,
        split_overlap=CHUNK_OVERLAP,
        split_respect_sentence_boundary=False,
    )

    chunks = chunker.process(docs)
    return processing.remove_identical_chunks(chunks)


def preselect_similar_chunks(doc_chunks):
    print("[▶] Selecting semantically similar chunks...")

    embeddings = processing.compute_chunk_embeddings(
        doc_chunks, EMBEDDING_MODEL_NAME, show_progress_bar=True
    )

    for chunk, emb in zip(doc_chunks, embeddings):
        chunk.embedding = emb

    similarity = processing.compute_chunk_similarity(doc_chunks)
    index_pairs = processing.get_top_n_similar_chunk_pair_indices(
        similarity, TOP_N_SIMILAR_CHUNKS
    )

    chunk_ids = [c.id for c in doc_chunks]
    id_pairs = [(chunk_ids[i], chunk_ids[j]) for i, j in index_pairs]

    return {c.id: c for c in doc_chunks}, id_pairs


def find_contradictions(chunks, chunk_pairs):
    print("[▶] Detecting sentence-level contradictions...")

    tokenizer, model = scoring.load_contradiction_model()
    rows = []

    for id_a, id_b in chunk_pairs:
        sents_a = split_into_sentences(chunks[id_a].content)
        sents_b = split_into_sentences(chunks[id_b].content)

        if not sents_a or not sents_b:
            continue

        emb_a = compute_sentence_embeddings(sents_a, EMBEDDING_MODEL_NAME)
        emb_b = compute_sentence_embeddings(sents_b, EMBEDDING_MODEL_NAME)

        sim = cosine_similarity(emb_a, emb_b)

        for i in range(len(sents_a)):
            for j in range(len(sents_b)):
                if sim[i][j] >= SENTENCE_SIMILARITY_THRESHOLD:
                    entail, neutral, contra = scoring._evaluate_contradictions(
                        sents_a[i], sents_b[j], tokenizer, model
                    )

                    rows.append([
                        id_a, id_b, i, j,
                        entail, neutral, contra,
                        sents_a[i], sents_b[j]
                    ])

    return pd.DataFrame(
        rows,
        columns=[
            "chunk_A", "chunk_B",
            "sentence_A", "sentence_B",
            "entailment", "neutral", "contradiction",
            "sentence_A_text", "sentence_B_text"
        ]
    )


if __name__ == "__main__":
    df = load_documents()
    chunks = preprocess_documents(df)
    chunk_map, chunk_pairs = preselect_similar_chunks(chunks)

    candidates = find_contradictions(chunk_map, chunk_pairs)

    os.makedirs("output", exist_ok=True)
    candidates.to_csv("output/candidates.csv", index=False)

    print("[▶] Running evaluation...")
    run_evaluation()
