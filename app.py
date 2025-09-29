import gradio as gr
from fastapi import FastAPI
import torch
from PIL import Image
import io
from ultralytics import YOLO
import requests  # Pra suportar file upload via API

# Carregar modelo
model = YOLO("Orientation.pt")

# Classes
CLASS_NAMES = {0: "Prints", 1: "4Flats"}

# Função de predição (igual ao server.py)
def predict_image(image):
    if image is None:
        return {"predictions": []}
    
    with torch.no_grad():
        results = model(image, verbose=False, conf=0.25, imgsz=640)
    
    predictions = []
    if results and results[0].obb:
        obb = results[0].obb
        for i in range(len(obb)):
            points = obb.xyxyxyxy[i].flatten().tolist()
            class_id = int(obb.cls[i].item())
            score = float(obb.conf[i].item())
            predictions.append({
                "label": CLASS_NAMES.get(class_id, str(class_id)),
                "points": points,
                "type": "polygon",
                "score": score
            })
    return {"predictions": predictions}

# Interface Gradio (pra demo visual)
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(label="Predições (JSON pro CVAT)"),
    title="Orientation YOLOv11x-OBB Detector",
    description="Faça upload de uma imagem pra detectar Prints/4Flats com OBB polygons."
)

# API endpoint (pra CVAT chamar)
fastapi_app = FastAPI()
@fastapi_app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict_image(image)

# Mount Gradio no FastAPI
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, app=fastapi_app)
