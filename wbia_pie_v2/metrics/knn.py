# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors


def predict_k_neigh(db_emb, db_lbls, test_emb, k=5):
    """Predict k nearest solutions for test embeddings based on labelled database embeddings.
    Input:
    db_emb: 2D float array (num_emb, emb_size): database embeddings
    db_lbls: 1D array, string or floats: database labels
    test_emb: 2D float array: test embeddings
    k: integer, number of predictions.
    Returns:
    neigh_lbl_un - 2d int array of shape [len(test_emb), k] labels of predictions
    neigh_ind_un - 2d int array of shape [len(test_emb), k] labels of indices of nearest points
    neigh_dist_un - 2d float array of shape [len(test_emb), k] distances of predictions
    """
    # Set number of nearest points (with duplicated labels)
    k_w_dupl = min(50, len(db_emb))
    nn_classifier = NearestNeighbors(n_neighbors=k_w_dupl, metric='euclidean')
    nn_classifier.fit(db_emb, db_lbls)

    # Predict nearest neighbors and distances for test embeddings
    neigh_dist, neigh_ind = nn_classifier.kneighbors(test_emb)

    # Get labels of nearest neighbors
    neigh_lbl = np.zeros(shape=neigh_ind.shape, dtype=db_lbls.dtype)
    for i, preds in enumerate(neigh_ind):
        for j, pred in enumerate(preds):
            neigh_lbl[i, j] = db_lbls[pred]

    # Remove duplicates
    neigh_lbl_un = []
    neigh_ind_un = []
    neigh_dist_un = []

    for j in range(neigh_lbl.shape[0]):
        indices = np.arange(0, len(neigh_lbl[j]))
        a, b = rem_dupl(neigh_lbl[j], indices)
        neigh_lbl_un.append(a[:k])
        neigh_ind_un.append(neigh_ind[j][b][:k].tolist())
        neigh_dist_un.append(neigh_dist[j][b][:k].tolist())

    return neigh_lbl_un, neigh_ind_un, neigh_dist_un


def pred_light(query_embedding, db_embeddings, db_labels, n_results=10):
    # Fit nearest neighbours classifier
    neigh_lbl_un, neigh_ind_un, neigh_dist_un = predict_k_neigh(
        db_embeddings, db_labels, query_embedding, k=n_results
    )

    neigh_lbl_un = neigh_lbl_un[0]
    neigh_dist_un = neigh_dist_un[0]

    ans_dict = [
        {'label': lbl, 'distance': dist} for lbl, dist in zip(neigh_lbl_un, neigh_dist_un)
    ]
    return ans_dict


def rem_dupl(seq, seq2=None):
    """Remove duplicates from a sequence and keep the order of elements. Do it in unison with a sequence 2."""
    seen = set()
    seen_add = seen.add
    if seq2 is None:
        return [x for x in seq if not (x in seen or seen_add(x))]
    else:
        a = [x for x in seq if not (x in seen or seen_add(x))]
        seen = set()
        seen_add = seen.add
        b = [seq2[i] for i, x in enumerate(seq) if not (x in seen or seen_add(x))]
        return a, b
