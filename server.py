# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from torchvision import transforms
import io
from ultralytics.nn.tasks import OBBModel  # sua classe personalizada

app = FastAPI(title="Orientation Model Serverless")

# Classes do modelo
CLASS_NAMES = {
    0: "Prints",
    1: "4Flats"
}

# Carregar modelo de forma segura
with torch.serialization.add_safe_globals([OBBModel]):
    model = torch.load("Orientation.pt", weights_only=True)
model.eval()

# Transformação padrão (ajuste se seu modelo exigir algo diferente)
preprocess = transforms.Compose([
    transforms.ToTensor(),  # converte para tensor CxHxW
    # Adicione normalização se seu modelo exigir
])

def format_cvat_output(predictions):
    """
    predictions: lista de detecções [x1, y1, x2, y2, class_id, score]
    """
    results = []
    for det in predictions:
        x1, y1, x2, y2, class_id, score = det
        results.append({
            "label": CLASS_NAMES.get(int(class_id), str(int(class_id))),
            "points": [x1, y1, x2, y2],
            "type": "rectangle",
            "score": float(score)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    image_tensor = preprocess(image).unsqueeze(0)  # adiciona batch dimension

    # Inferência
    with torch.no_grad():
        outputs = model(image_tensor)
        # Se tensor, converta para lista
        if torch.is_tensor(outputs):
            outputs = outputs.tolist()

    # Formatar para CVAT
    response = {"predictions": format_cvat_output(outputs)}

    return JSONResponse(content=response)
