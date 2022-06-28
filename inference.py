# from src.feature_extract import Img2VecConvNext
from src.feature_embedding import FeatureEmbedding
from numpy import dot
from numpy.linalg import norm
import os
import pickle
import torch
from PIL import Image
from torchvision import transforms
import io
import matplotlib.pyplot as plt

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def embed2dict(data_path, mode='normal'):
    embeddings = {}
    for folder in os.listdir(data_path):
        for file in os.listdir(data_path + folder):
            img_path = data_path + folder + "/" + file
            img = preprocess_img(img_path, mode)
            img_embedding = feature_embedding(img).cpu().detach().numpy()
            save_path = folder + "/" + file
            embeddings[save_path] = img_embedding
    return embeddings

def preprocess_img(img_path, mode):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'api':
        img = Image.open(io.BytesIO(img_path))
    else:
        img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def get_similarities(embeddings, img_path, mode):
    img = preprocess_img(img_path, mode)
    input_embedding = feature_embedding(img).cpu().detach().numpy()
    similarities = {}
    for key, value in embeddings.items():
        similarities[key] = cosine_similarity(input_embedding, value)
    return similarities

def get_similar_images(embeddings, img_path, mode, topk=10):
    similarities = get_similarities(embeddings, img_path, mode)
    sorted_similarities = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_similarities[:topk]

def plot_output_subplot(similarities):
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    data_dir = "data/animals"
    for i, (key, value) in enumerate(similarities):
        img_path = data_dir + "/" + key
        img = Image.open(img_path)
        axs[i // 5, i % 5].imshow(img)
        axs[i // 5, i % 5].set_title(value)
        axs[i // 5, i % 5].axis('off')
    plt.show()

if __name__ == "__main__":
    feature_embedding = FeatureEmbedding()
    
    with open('./models/animals_effnetb0.pkl', 'rb') as f:
        animals_embeddings = pickle.load(f)

    similars = get_similar_images(animals_embeddings, './examples/kong_2.jpg', 'file', topk=10)
    print(similars)
    plot_output_subplot(similars)
    