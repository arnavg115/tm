from typing import List, Union, Dict
from utils import get_embeddings, dim_reduc, clustering, find_describer, parser
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
    body: Union[List[List[str]], List[Dict[str, str]]]
    labels: Union[List[str], None] = [""]
    merge_str: Union[str, None] = "default_def"
    parse: bool = False
    model: str = "sentence-transformers/all-distilroberta-v1"
    describers: bool = False


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/v1/tms")
def topic_modeling(q: query):
    corpus = []
    body = q.body
    if q.parse:
        body = parser(q.body)
    if len(q.labels) > 1:
        for qy in body:
            st = merge[q.merge_str] if q.merge_str in merge.keys() else q.merge_str
            for i, label in enumerate(q.labels):
                st = st.replace(f"[{label}]", qy[i])
            corpus.append(st)
    else:
        corpus = q.body
    # return corpus

    done = False
    j = 0
    while j < 5 and not done:
        embeddings = get_embeddings(corpus, config["HF"], model=q.model)
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
        labs = np.array(body)[clustered == cluster]
        if q.describers:
            describers = find_describer(sentences).tolist()
            out.append({"body": labs, "describers": describers})
        else:
            out.append({"body": labs})
    return out
