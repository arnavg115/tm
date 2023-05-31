from functools import reduce
from typing import List
import requests
import json
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering,OPTICS, KMeans
from stopwords import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_embeddings(docs: List[str], hf_key: str, model:str = "sentence-transformers/all-distilroberta-v1"):
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/{}".format(model)
    headers = {"Authorization": f"Bearer {hf_key}"}
    data = json.dumps({"inputs":docs})
    out = requests.post(API_URL, data=data,headers=headers).json()
    return np.array(out)

def dim_reduc(embeddings: np.ndarray):
    # print(embeddings.shape)
    model = TSNE(perplexity=1, early_exaggeration=2)
    reduc = model.fit_transform(embeddings)
    return reduc

def clustering(data: np.ndarray):
    clusterer = OPTICS(min_samples=2)
    labels = clusterer.fit_predict(data)
    return np.array(labels)

def vocab_builder(text):
    regex = r"[().?!]"
    depunct = re.sub(r"\[.*\]","",re.sub(regex, "", text))
    destop = [word for word in depunct.lower().split() if word not in stopwords]
    return list(set(destop))


def unique(words: List[str]):
    l = set()
    for word in words:
        l.add(word)

def find_describer_bert(text_embedding: np.ndarray, vocab: List[str],hf_key:str, n:int = 5):
    word_embeddings = get_embeddings(vocab, hf_key)
    dist = np.array([distance.cosine(vec, text_embedding) for vec in word_embeddings])
    return np.array(vocab)[np.argsort(dist)[:n]]

def find_describer(text:str):

    vectorizer = TfidfVectorizer()
    destop = " ".join([word for word in text.lower().split() if word not in stopwords])
    out = vectorizer.fit_transform(destop.split("."))
    vocab = vectorizer.get_feature_names_out()
    return vocab[np.argsort(np.mean(out, axis=0)[0])[0,-10:]][0]