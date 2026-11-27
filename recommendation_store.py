import json
from pathlib import Path

import numpy as np
from qdrant_client import models

from build_index_data import normalize, read_rows
from features import Encoder, histories

ROOT = Path(__file__).resolve().parent
COLLECTION = "netflix_titles"


class Store:
    def __init__(self, client):
        self.client = client
        self.catalog = json.loads((ROOT / "serving/catalog.json").read_text())
        self.history = histories(read_rows(ROOT / "data/clean/train.csv"))
        with np.load(ROOT / "serving/vectors.npz") as weights:
            self.vectors = weights["titles"]
            self.mw1, self.mb1, self.mw2 = (weights[k] for k in ("mw1", "mb1", "mw2"))
            self.encoder = Encoder.from_features(
                [row["title_id"] for row in self.catalog], weights["features"],
                [tuple(pair) for pair in weights["context"].tolist()])
        self.index_titles()

    def index_titles(self):
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(
                size=self.vectors.shape[1], distance=models.Distance.COSINE))
        points = [models.PointStruct(id=int(row["title_id"][3:]), vector=vector.tolist(), payload=row)
                  for row, vector in zip(self.catalog, self.vectors)]
        self.client.upsert(COLLECTION, points=points, wait=True)
        if self.client.count(COLLECTION, exact=True).count != len(self.catalog):
            raise ValueError("Qdrant collection does not match the catalog")

    def picks(self, member):
        history = self.history.get(member)
        if history is None:
            return []
        indices = [self.encoder.index[r["title_id"]] for r in history if r["title_id"] in self.encoder.index]
        features = self.encoder.member(history)
        taste = normalize(np.tanh(features @ self.mw1 + self.mb1) @ self.mw2)
        descriptions = self.vectors[:, len(taste):]
        text = normalize(descriptions[indices].mean(axis=0)) if indices else np.zeros(descriptions.shape[1])
        query = np.concatenate((taste, text)).tolist()
        seen = sorted({int(self.encoder.ids[i][3:]) for i in indices})
        exclude = models.Filter(must_not=[models.HasIdCondition(has_id=seen)]) if seen else None
        hits = self.client.query_points(COLLECTION, query=query, query_filter=exclude,
                                        limit=10, with_payload=True).points
        return [dict(hit.payload, score=hit.score) for hit in sorted(hits, key=lambda h: (-h.score, h.id))]
