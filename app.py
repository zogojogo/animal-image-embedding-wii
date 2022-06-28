from fastapi import FastAPI, File, UploadFile
from starlette.responses import Response
import uvicorn
from inference import get_similar_images, preprocess_img, get_similarities, cosine_similarity
from src.feature_embedding import FeatureEmbedding
from torch.nn import functional as F
import time
import pickle

feature_embedding = FeatureEmbedding()
with open('animals_effnetb0.pkl', 'rb') as f:
    animals_embeddings = pickle.load(f)

# Create Fast API
app = FastAPI()

@app.get("/")
async def index():
    return {"messages": "Open the documentations /docs or /redoc"}

@app.post("/animals_embedding")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        start_time = time.time()
        img = preprocess_img(image, 'api')
        img_embedding = feature_embedding(img).cpu().detach().numpy()
        similarities = {}
        for key, value in animals_embeddings.items():
            similarities[key] = cosine_similarity(img_embedding, value)
        sorted_similarities = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
        # similars = get_similar_images(animals_embeddings, image, 'api', topk=10)
        end_time = time.time()
        return {
            "input filename": str(file.filename),
            "similar image": str(sorted_similarities[0][0]),
            "similarity": str(sorted_similarities[0][1]),
            "inference time": str(end_time - start_time)
        }
    except:
        return Response("Internal server error", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
