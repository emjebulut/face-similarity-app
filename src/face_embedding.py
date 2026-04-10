import numpy as np
import torch
from PIL import image
from facenet_pytorch import MCNN , InceptionResnetV1
from src.config import DEVICE

mtcnn = MTCNN( image_size = 160 , margin = 20 , keep_all = false , device = DEVICE )
resnet = InceptioResnetV1( pretrained = "vggface2").eval().to(DEVICE)

def get_face_embedding( img = Image.Image ) -> np.ndarray:
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected in the image")
    face = face.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(face)
    emb = emb.squeeze(0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

