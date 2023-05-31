from typing import List
from utils import get_embeddings, dim_reduc, clustering, find_describer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
from dotenv import dotenv_values
import time

config = {
    **dotenv_values(".env.local"),
    **os.environ,  # load shared development variables
}
app = FastAPI()

merge = {"default_def": "[term] means [definition]."}


class query(BaseModel):
    body: List[List[str]] | List[str]
    labels: List[str] | None = [""]
    merge_str: str | None = "default_def"
    embeddings: List[List[float]] | None = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/v1/tms")
def topic_modeling(q: query):
    corpus = []
    if len(q.labels) > 1:
        for qy in q.body:
            st = merge[q.merge_str] if q.merge_str in merge.keys() else q.merge_str
            for i, label in enumerate(q.labels):
                st = st.replace(f"[{label}]", qy[i])
            corpus.append(st)
    else:
        corpus = q.body
    # return corpus
    embeddings = q.embeddings
    if q.embeddings is None:
        done = False
        j = 0
        while j < 5 and not done:
            embeddings = get_embeddings(corpus, config["HF_KEY"])
            j += 1
            done = type(embeddings) is np.ndarray
            time.sleep(1)
        if not done:
            raise HTTPException(500, "Embeddings not loaded")

    red = dim_reduc(embeddings)
    clustered = clustering(red)
    out = []
    for cluster in np.unique(clustered):
        sentences = ".".join(np.array(corpus)[clustered == cluster])
        labs = np.array(q.body)[clustered == cluster]
        describers = find_describer(sentences).tolist()
        out.append({"labels": labs, "describers": describers})
    return out
