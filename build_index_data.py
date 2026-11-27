"""Export catalog embeddings and the inference weights from the trained checkpoint."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from features import Encoder


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def description_features(descriptions):
    """TF-IDF followed by a deterministic 64-dimensional latent semantic projection."""
    documents = [Counter(re.findall(r"[a-z]{3,}", text.lower())) for text in descriptions]
    if not all(documents):
        raise ValueError("Every catalog row needs a description")
    counts = Counter(word for document in documents for word in document)
    vocabulary = {word: i for i, word in enumerate(sorted(counts))}
    features = np.zeros((len(documents), len(vocabulary)))
    for i, document in enumerate(documents):
        for word, freq in document.items():
            features[i, vocabulary[word]] = (1 + np.log(freq)) * np.log((1 + len(documents)) / (1 + counts[word]))
    features = normalize(features)
    values, axes = np.linalg.eigh(features @ features.T)
    return normalize(axes[:, -64:] * np.sqrt(np.maximum(values[-64:], 0)))


def export(catalog, train, checkpoint, output):
    catalog = sorted(catalog, key=lambda row: row["title_id"])
    encoder = Encoder(train, catalog)
    if encoder.ids != [f"TT-{i:04}" for i in range(1, 301)]:
        raise ValueError("Expected every catalog title TT-0001 through TT-0300 exactly once")
    if encoder.ids != checkpoint["title_ids"].tolist():
        raise ValueError("Catalog differs from the training checkpoint")
    text = description_features([row["description"] for row in catalog])
    # Two content channels: the trained metadata tower and description semantics.
    latent = np.tanh(encoder.titles @ checkpoint["tw1"] + checkpoint["tb1"]) @ checkpoint["tw2"]
    title_vectors = np.column_stack((normalize(latent), text))
    output.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(output / "vectors.npz", titles=title_vectors.astype("float32"),
                        features=encoder.titles.astype("float32"), context=np.array(encoder.context),
                        **{k: checkpoint[k] for k in ("mw1", "mb1", "mw2")})
    fields = ("title_id", "title_name", "primary_genre")
    metadata = [{k: row[k] for k in fields} for row in catalog]
    (output / "catalog.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--catalog", required=True)
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--train", default="data/clean/train.csv")
    cli.add_argument("--out", type=Path, default=Path("serving"))
    args = cli.parse_args()
    with np.load(args.checkpoint) as weights:
        export(read_rows(args.catalog), read_rows(args.train), weights, args.out)
