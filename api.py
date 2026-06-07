from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import uuid
import os
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://car-damage-web.vercel.app",
        "https://car-damage-eg5p6iv3i-alfredo-reyes-projects1809.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models_trained/A/best.pt")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    results = model.predict(
        source=image_path,
        conf=0.55,
        save=False,
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist(),
            }
        )

    annotated_image = result.plot()

    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated_image)

    try:
        os.remove(image_path)
    except OSError:
        pass

    return {
        "detections": detections,
        "annotated_image_url": f"/outputs/{output_filename}",
    }