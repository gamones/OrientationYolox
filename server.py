# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from ultralytics.nn.tasks import OBBModel  # sua classe personalizada

app = FastAPI(title="Orientation Model Serverless")

# Carregar modelo de forma segura
with torch.serialization.add_safe_globals([OBBModel]):
    model = torch.load("Orientation.pt", weights_only=True)
model.eval()

# Função auxiliar para converter saída do modelo para CVAT JSON
def format_cvat_output(predictions):
    """
    predictions: saída do seu modelo, lista de detecções
    Cada detecção: [x1, y1, x2, y2, class_id, score]
    """
    results = []
    for det in predictions:
        x1, y1, x2, y2, class_id, score = det
        results.append({
            "label": str(int(class_id)),
            "points": [x1, y1, x2, y2],
            "type": "rectangle",
            "score": float(score)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem enviada
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Aqui você aplica seu preprocessing se houver (transform)
    # Exemplo genérico: transform = ...
    # image_tensor = transform(image).unsqueeze(0)

    # Inferência
    with torch.no_grad():
        outputs = model(image)  # adapte se seu modelo exigir tensor 4D
        # Se a saída for tensor, converta para lista
        if torch.is_tensor(outputs):
            outputs = outputs.tolist()

    # Formatar para JSON CVAT
    response = {"predictions": format_cvat_output(outputs)}

    return JSONResponse(content=response)
