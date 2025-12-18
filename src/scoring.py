import itertools
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize


def load_contradiction_model(
    model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _evaluate_contradictions(premise, hypothesis, tokenizer, model):
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0].tolist()
    return probs  # entailment, neutral, contradiction


def compute_sentence_contradiction_scores(
    chunks, chunk_pairs, tokenizer, model
):
    rows = []

    for id_a, id_b in chunk_pairs:
        chunk_a = chunks[id_a]
        chunk_b = chunks[id_b]

        sents_a = sent_tokenize(chunk_a.content)
        sents_b = sent_tokenize(chunk_b.content)

        for i, j in itertools.product(
            range(len(sents_a)), range(len(sents_b))
        ):
            entail, neutral, contra = _evaluate_contradictions(
                sents_a[i], sents_b[j], tokenizer, model
            )

            rows.append(
                [id_a, id_b, i, j, entail, neutral, contra]
            )

    return pd.DataFrame(
        rows,
        columns=[
            "chunk_A",
            "chunk_B",
            "sentence_A",
            "sentence_B",
            "entailment",
            "neutral",
            "contradiction",
        ],
    )


def get_top_k_contradictive_candidates(df, k):
    return df.sort_values(
        "contradiction", ascending=False
    ).head(k)


def retrieve_candidate_info(candidates, chunks):
    candidates = candidates.copy()

    candidates["sentence_A_text"] = [
        sent_tokenize(chunks[c].content)[i]
        for c, i in zip(candidates.chunk_A, candidates.sentence_A)
    ]

    candidates["sentence_B_text"] = [
        sent_tokenize(chunks[c].content)[i]
        for c, i in zip(candidates.chunk_B, candidates.sentence_B)
    ]

    return candidates
