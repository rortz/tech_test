import os
import json
import logging
import swifter
import numpy as np
import pandas as pd

from scipy import sparse
from collections import defaultdict
from scipy.spatial import cKDTree as KDTree

from gensim import utils, models

class DataManager:
    
    def text_tokenizer(self, items_titles, stopwords):
        treat_list_docs = []

        for doc in items_titles:
            clean_tokenized_text = utils.simple_preprocess(doc, deacc=True)
            clean_tokenized_text_sw = [w for w in clean_tokenized_text if w not in stopwords]

            if len(clean_tokenized_text_sw) > 1:
                treat_list_docs.append(clean_tokenized_text_sw)

        return treat_list_docs

    def set_configurations(self, config_name):
        config = json.load(open(f'{config_name}.json'))
        logging.info(f"Configuration file {config_name} loaded successfully")

        return config

    def kdtree_similarity(self, df, feature="product_name"):
        df['embeddings_vector'] = df['embeddings_vector'].swifter.apply(lambda x: np.array(x))
        norm = df['embeddings_vector'].swifter.apply(
                lambda x: np.linalg.norm(x, axis=0))
        df['embeddings_vector'] = df['embeddings_vector']/norm

        A = KDTree(df['embeddings_vector'].to_list())
        D = A.sparse_distance_matrix(A, 1, p=2.0, output_type='ndarray')
        
        DU = D[D['i'] < D['j']]

        coo_sparse_matrix = sparse.coo_matrix((DU['v'], (DU['i'], DU['j'])))

        def sort_coo(m):
            tuples = zip(m.row, m.col, m.data)
            
            return sorted(tuples, key=lambda x: (x[0], x[2]))

        sorted_rows = defaultdict(list)
        for i in sort_coo(coo_sparse_matrix):
            sorted_rows[i[0]].append((i[1], i[2]))
            sorted_rows[i[1]].append((i[0], i[2]))

        named_rows = defaultdict(list)
        col_idx = df.columns.get_loc(feature)
        for item_key, pairs in sorted_rows.items():
            item_name = df.iat[item_key, col_idx]
            for pair in pairs[:3]: #top3
                other_key = pair[0]
                other_name = df.iat[other_key, col_idx]
                named_rows[item_name].append(other_name)
        
        return named_rows, DU

