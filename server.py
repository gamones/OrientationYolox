# server.py
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
from ultralytics.nn.tasks import OBBModel  # importa a classe do seu modelo

app = FastAPI()

# Carregar modelo de forma segura
with torch.serialization.add_safe_globals([OBBModel]):
    model = torch.load("Orientation.pt", weights_only=True)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess conforme seu modelo
    # Exemplo genérico:
    # transform = ...
    # image = transform(image).unsqueeze(0)
    
    # Inferência
    with torch.no_grad():
        outputs = model(image)  # adapte conforme sua saída

    # Retornar JSON no formato que CVAT espera
    return {"predictions": outputs.tolist()}  # adapte para boxes, classes etc.
