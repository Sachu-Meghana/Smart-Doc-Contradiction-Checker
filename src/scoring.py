import itertools
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from src.processing import split_into_sentences


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
