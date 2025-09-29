# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from ultralytics.nn.tasks import OBBModel  # classe personalizada do seu modelo

app = FastAPI(title="Orientation Model Serverless")

# Mapear IDs para nomes de classes
CLASS_NAMES = {
    0: "Prints",
    1: "4Flats"
}

# Carregar modelo de forma segura
with torch.serialization.add_safe_globals([OBBModel]):
    model = torch.load("Orientation.pt", weights_only=True)
model.eval()

def format_cvat_output(predictions):
    """
    predictions: lista de detecções do modelo
    Cada detecção: [x_center, y_center, width, height, angle, class_id, score]
    """
    results = []
    for det in predictions:
        x_c, y_c, w, h, angle, class_id, score = det

        results.append({
            "label": CLASS_NAMES.get(int(class_id), str(int(class_id))),
            "points": [x_c, y_c, w, h, angle],  # OBB points
            "type": "obb",  # informa ao CVAT que é OBB
            "score": float(score)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem enviada
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Aqui você aplica seu preprocessing se houver
    # Exemplo genérico: transform = ...
    # image_tensor = transform(image).unsqueeze(0)

    # Inferência
    with torch.no_grad():
        outputs = model(image)  # se precisar, adapte para tensor 4D
        if torch.is_tensor(outputs):
            outputs = outputs.tolist()

    # Formatar para JSON que CVAT espera
    response = {"predictions": format_cvat_output(outputs)}
    return JSONResponse(content=response)
