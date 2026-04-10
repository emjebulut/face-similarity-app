import os
import numpy as np
from PIL import Image

from src.face_embedding import get_face_embedding


def load_reference_embeddings(base_dir="data/celebrities"):
    refs = {}
    if not os.path.isdir(base_dir):
        return refs
    for person in os.listdir(base_dir):
        person_dir = os.path.join( base_dir , person )
        if not os.path.isdir(person_dir):
            continue
        embs = []
        for file in os.listdir(person_dir):
            if file.lower().endswith(( ".jpg" , ".jpeg" , ".png" )):
                try:
                    img = Image.open(os.path.join(person_dir , file )).convert("RGB")
                    embs.append(get_face_embedding(img))
                except Expection:
                    continue
        if embs:
            refs[person] = np.mean( embs , axis=0 )
    return refs