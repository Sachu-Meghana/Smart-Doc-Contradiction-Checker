from functools import partial
from typing import Dict, List, Tuple

from haystack.schema import Document
from pandas import DataFrame

from src import custom_preprocessors, loading, processing, scoring
from config import *


def load_documents() -> DataFrame:
    print("[▶] Loading data...")
    df = loading.load_dataset_from_json(DATASET_FILEPATH)
    print("[✓] Finished loading!")
    return df


def preprocess_documents(df: DataFrame) -> List[Document]:
    print("[▶] Processing data...")
    df["fulltext"] = df.text_by_page.apply(processing.clean_and_combine_pages)

    if SUBSET_SIZE:
        df = df.iloc[:SUBSET_SIZE]

    docs = processing.convert_frame_to_haystack(df)

    print(" | [+] Splitting into chunks")

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

    doc_chunks = chunker.process(docs)
    doc_chunks = processing.remove_identical_chunks(doc_chunks)

    print("[✓] Finished processing!")
    return doc_chunks


def preselect_similar_chunks(
    doc_chunks: List[Document],
):
    print("[▶] Pre-selecting similar chunks...")
    print(" | [+] Computing chunk embeddings")

    embeddings = processing.compute_chunk_embeddings(
        doc_chunks, EMBEDDING_MODEL_NAME, show_progress_bar=True
    )

    for chunk, emb in zip(doc_chunks, embeddings):
        chunk.embedding = emb

    print(" | [+] Selecting similar chunks")
    similarity = processing.compute_chunk_similarity(doc_chunks)

    pairs_idx = processing.get_top_n_similar_chunk_pair_indices(
    similarity, TOP_N_SIMILAR_CHUNKS)

    chunks = {c.id: c for c in doc_chunks}

    pairs = [
        (doc_chunks[i].id, doc_chunks[j].id)
        for i, j in pairs_idx
    ]

    return chunks, pairs




def find_contradictions(
    chunks: Dict[str, Document],
    chunk_pairs,
):

    print("[▶] Selecting contradiction candidates...")
    print(" | [+] Loading models")

    tokenizer, model = scoring.load_contradiction_model()

    print(" | [+] Computing contradiction scores")

    scores = scoring.compute_sentence_contradiction_scores(
    chunks, chunk_pairs, tokenizer, model
)

    print(" | [+] Selecting candidates")

    top = scoring.get_top_k_contradictive_candidates(
        scores, TOP_K_CONTRADICTIONS
    )

    final = scoring.retrieve_candidate_info(top, chunks)

    return final


if __name__ == "__main__":
    df = load_documents()
    chunks = preprocess_documents(df)
    chunk_map, chunk_pairs = preselect_similar_chunks(chunks)
    candidates = find_contradictions(chunk_map, chunk_pairs, chunk_ids)


    candidates.to_csv("output/candidates.csv", index=False)
    print(".... Saving candidates to output/candidates.csv")
