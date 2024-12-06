from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import torch
from sentence_transformers import SentenceTransformer

def geometric_median(X, eps=1e-5):
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return torch.tensor(y1)

        y = y1

def tfidf_dist(documents, dist_type):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_matrix = np.array(tfidf_matrix.todense())
    tfidf_matrix = torch.tensor(tfidf_matrix)

    if dist_type == 'gm':
        gm = geometric_median(tfidf_matrix).unsqueeze(0)
        distance = torch.cdist(tfidf_matrix, gm, p=2).squeeze()
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")

    return distance

def sentence_bert(documents, model):
    model = SentenceTransformer(model)
    embeddings = model.encode(documents)
    
    return embeddings

def sbert_dist(documents, model, dist_type):
    # Get the sentence embeddings
    embeddings = sentence_bert(documents, model)
    embeddings = torch.tensor(embeddings).float()

    if dist_type == 'gm':
        gm = geometric_median(embeddings).unsqueeze(0).float()
        distance = torch.cdist(embeddings, gm, p=2).squeeze()
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")

    return distance