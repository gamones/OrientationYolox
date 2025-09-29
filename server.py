# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from ultralytics.nn.tasks import OBBModel  # sua classe personalizada
from torchvision import transforms

app = FastAPI(title="Orientation Model Serverless")

# Carregar modelo de forma segura
with torch.serialization.add_safe_globals([OBBModel]):
    model = torch.load("Orientation.pt", weights_only=True)
model.eval()

# Transformação de imagem (adapte tamanho/normalização se necessário)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # ajuste conforme seu modelo
    transforms.ToTensor()
])

def format_cvat_output(predictions):
    """
    predictions: lista de detecções do modelo
    Cada detecção: [x1, y1, x2, y2, x3, y3, x4, y4, class_id, score]
    """
    results = []
    for det in predictions:
        points = det[:8]  # 4 vértices: x1,y1,...,x4,y4
        class_id = int(det[8])
        score = float(det[9])
        results.append({
            "label": str(class_id),
            "points": points,
            "type": "obb",
            "score": score
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler imagem enviada
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # Inferência
    with torch.no_grad():
        outputs = model(image_tensor)  # saída: tensor [N, 10] (x1..x8, class, score)
        if torch.is_tensor(outputs):
            outputs = outputs.tolist()

    # Formatar para JSON CVAT
    response = {"predictions": format_cvat_output(outputs)}

    return JSONResponse(content=response)
