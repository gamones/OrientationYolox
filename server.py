from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from ultralytics import YOLO  # Use the YOLO wrapper for proper loading and inference

app = FastAPI(title="Orientation Model Serverless")

# Carregar modelo usando Ultralytics YOLO
model = YOLO("Orientation.pt")

# Classes do modelo
CLASS_NAMES = {
    0: "Prints",
    1: "4Flats"
}

# Converter saída para JSON compatível com CVAT
def format_cvat_output(results):
    """
    Extrai detecções OBB do results e formata para CVAT como polígonos.
    """
    predictions = []
    if results and results[0].obb:  # Verifica se há resultados OBB
        obb = results[0].obb
        for i in range(len(obb)):
            # Pontos do OBB como lista plana [x1, y1, x2, y2, x3, y3, x4, y4]
            points = obb.xyxyxyxy[i].flatten().tolist()
            class_id = int(obb.cls[i].item())
            score = float(obb.conf[i].item())
            predictions.append({
                "label": CLASS_NAMES.get(class_id, str(class_id)),
                "points": points,
                "type": "polygon",  # Usando polygon para OBB
                "score": score
            })
    return predictions

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Rodar inferência usando o método predict do YOLO
    with torch.no_grad():
        results = model(image, verbose=False)  # results é uma lista de Results objects

    # Resposta no formato CVAT
    return JSONResponse(content={"predictions": format_cvat_output(results)})
