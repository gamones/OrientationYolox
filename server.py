# server.py
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Carregar modelo
model = torch.load("Orientation.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess conforme seu modelo
    # Exemplo genérico: transformações do PyTorch
    # transform = ...
    # image = transform(image).unsqueeze(0)
    
    # Inferência
    with torch.no_grad():
        outputs = model(image)  # adapte conforme sua saída

    # Retornar JSON no formato CVAT espera
    return {"predictions": outputs.tolist()}  # você adapta para boxes, classes etc.
