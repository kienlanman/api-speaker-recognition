
from src.eval_metrics import *
import keras.backend as K

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)
    return s

# def batch_cosine_similarity(x1, x2):
#     # https://en.wikipedia.org/wiki/Cosine_similarity
#     # 1 = equal direction ; -1 = opposite direction
#     dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
#     return dot