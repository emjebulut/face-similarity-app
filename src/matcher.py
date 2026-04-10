import numpy as np


def cosine_similarity( a : np.ndarray , b : np.ndarray ) -> float:
    return float( np.dot( a , b ))


def find_top_k_matches(taget_embedding : np.ndarray , refs : dict , k : int = 5 ):
    scores = [ ( name , cosine_similarity( target_embedding , emb )) for name , emb in refs.items()]
    scores.sort( key = lambda x : x[1] , reverse=True )
    return scores[:k]
