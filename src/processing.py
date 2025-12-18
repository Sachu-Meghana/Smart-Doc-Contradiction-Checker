import numpy as np
import spacy
from haystack.schema import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def clean_and_combine_pages(pages):
    return "\n".join(pages)


def convert_frame_to_haystack(df):
    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                content=row.fulltext,
                meta={"title": row.title},
            )
        )
    return docs


def clean_sentence_splits(
    sentences,
    toc_period_threshold,
    length_minimum,
    length_maximum,
):
    cleaned = []
    for s in sentences:
        s = s.strip()
        if length_minimum <= len(s) <= length_maximum:
            cleaned.append(s)
    return cleaned


def remove_identical_chunks(chunks):
    seen = set()
    unique = []
    for c in chunks:
        if c.content not in seen:
            unique.append(c)
            seen.add(c.content)
    return unique


def compute_chunk_embeddings(chunks, model_name, show_progress_bar):
    model = SentenceTransformer(model_name)
    texts = [c.content for c in chunks]
    return model.encode(texts, show_progress_bar=show_progress_bar)


def compute_chunk_similarity(chunks):
    embeddings = np.array([c.embedding for c in chunks])
    return cosine_similarity(embeddings)


def get_top_n_similar_chunk_pair_indices(similarity, n):
    pairs = []
    for i in range(len(similarity)):
        for j in range(i + 1, len(similarity)):
            pairs.append((i, j, similarity[i][j]))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return [(p[0], p[1]) for p in pairs[:n]]


