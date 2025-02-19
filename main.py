from fastapi import FastAPI
from controllers.video_controller_hf import get_router
from transformers import AutoImageProcessor, Dinov2WithRegistersForImageClassification
import torch

# ref no momento: https://huggingface.co/WpythonW/dinoV2-deepfake-detector/tree/main

image_processor = AutoImageProcessor.from_pretrained('WpythonW/dinoV2-deepfake-detector')
model = Dinov2WithRegistersForImageClassification.from_pretrained("WpythonW/dinoV2-deepfake-detector")
model.config.id2label = {0: "FAKE", 1: "REAL"}
model.config.label2id = {"FAKE": 0, "REAL": 1}

app = FastAPI()

app.include_router(get_router(image_processor, model))

# test route
@app.get("/")
def home():
    return {"message": "API de Upload de Vídeos está funfando!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
