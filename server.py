# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from ultralytics.nn.tasks import OBBModel  # classe usada no seu modelo

app = FastAPI(title="Orientation Model Serverless")

# Registrar a classe customizada para permitir deserialização
torch.serialization.add_safe_globals([OBBModel])

# Carregar modelo
model = torch.load("Orientation.pt", weights_only=True)
model.eval()

# Classes do modelo
CLASS_NAMES = {
    0: "Prints",
    1: "4Flats"
}

# Converter saída para JSON compatível com CVAT
def format_cvat_output(predictions):
    """
    predictions: lista de detecções
    Cada detecção deve ser: [x1, y1, x2, y2, class_id, score]
    """
    results = []
    for det in predictions:
        x1, y1, x2, y2, class_id, score = det
        results.append({
            "label": CLASS_NAMES.get(int(class_id), str(int(class_id))),
            "points": [x1, y1, x2, y2],
            "type": "rectangle",  # pode trocar para "polygon" se quiser OBB poligonal
            "score": float(score)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO: aplicar transforms conforme treino (resize, normalize etc.)
    # Exemplo genérico:
    # transform = ...
    # image_tensor = transform(image).unsqueeze(0)

    # Rodar inferência
    with torch.no_grad():
        outputs = model(image)  # ajuste para formato esperado pelo seu modelo
        if torch.is_tensor(outputs):
            outputs = outputs.tolist()

    # Resposta no formato CVAT
    return JSONResponse(content={"predictions": format_cvat_output(outputs)})
