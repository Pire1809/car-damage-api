from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import uuid
import os
import cv2
from datetime import datetime

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


REPAIR_COSTS = {
    "scratch": {
        "low": (150, 350),
        "medium": (350, 900),
        "high": (900, 2500),
    },
    "dent": {
        "low": (200, 600),
        "medium": (600, 1500),
        "high": (1500, 3500),
    },
    "crack": {
        "low": (300, 800),
        "medium": (800, 2000),
        "high": (2000, 5000),
    },
    "glass_shatter": {
        "low": (250, 600),
        "medium": (600, 1500),
        "high": (1500, 3000),
    },
    "lamp_broken": {
        "low": (250, 700),
        "medium": (700, 1800),
        "high": (1800, 4000),
    },
    "tire_flat": {
        "low": (100, 300),
        "medium": (300, 700),
        "high": (700, 1500),
    },
}


BRAND_MULTIPLIERS = {
    "toyota": 1.00,
    "honda": 1.00,
    "nissan": 1.00,
    "hyundai": 0.95,
    "kia": 0.95,
    "ford": 1.05,
    "chevrolet": 1.05,
    "dodge": 1.10,
    "jeep": 1.15,
    "subaru": 1.10,
    "volkswagen": 1.15,
    "mazda": 1.05,
    "tesla": 1.60,
    "bmw": 1.55,
    "mercedes-benz": 1.65,
    "mercedes": 1.65,
    "audi": 1.60,
    "lexus": 1.35,
    "acura": 1.25,
    "infiniti": 1.25,
    "volvo": 1.40,
    "porsche": 2.00,
    "land rover": 1.90,
    "range rover": 1.90,
}


VEHICLE_TYPE_MULTIPLIERS = {
    "compact": 0.90,
    "sedan": 1.00,
    "suv": 1.15,
    "pickup": 1.20,
    "luxury": 1.45,
    "ev": 1.55,
    "hybrid": 1.25,
    "performance": 1.80,
    "exotic": 2.20,
}


def normalize_text(value: str) -> str:
    return value.strip().lower() if value else ""


def get_damage_severity(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def get_year_multiplier(year: int | None) -> float:
    if not year:
        return 1.00

    current_year = datetime.now().year
    age = current_year - year

    if age <= 3:
        return 1.25
    if age <= 7:
        return 1.10
    if age <= 12:
        return 1.00
    return 0.90


def get_detection_count_multiplier(count: int) -> float:
    if count <= 1:
        return 1.00
    if count == 2:
        return 1.15
    if count == 3:
        return 1.30
    return 1.50


def calculate_repair_estimate(
    detections: list[dict],
    make: str,
    vehicle_type: str,
    year: int | None,
) -> dict:
    if not detections:
        return {
            "min": 0,
            "max": 0,
            "currency": "USD",
            "severity": "None",
            "summary": "No visible vehicle damage was detected.",
            "disclaimer": "This is an AI-generated preliminary estimate and not an official repair quote.",
        }

    base_min = 0
    base_max = 0
    severity_scores = []

    for detection in detections:
        damage_type = detection["class_name"]
        confidence = detection["confidence"]
        severity = get_damage_severity(confidence)

        if damage_type not in REPAIR_COSTS:
            continue

        cost_min, cost_max = REPAIR_COSTS[damage_type][severity]

        base_min += cost_min
        base_max += cost_max

        if severity == "high":
            severity_scores.append(3)
        elif severity == "medium":
            severity_scores.append(2)
        else:
            severity_scores.append(1)

    make_key = normalize_text(make)
    vehicle_type_key = normalize_text(vehicle_type)

    brand_multiplier = BRAND_MULTIPLIERS.get(make_key, 1.10)
    vehicle_type_multiplier = VEHICLE_TYPE_MULTIPLIERS.get(vehicle_type_key, 1.00)
    year_multiplier = get_year_multiplier(year)
    count_multiplier = get_detection_count_multiplier(len(detections))

    final_min = (
        base_min
        * brand_multiplier
        * vehicle_type_multiplier
        * year_multiplier
        * count_multiplier
    )
    final_max = (
        base_max
        * brand_multiplier
        * vehicle_type_multiplier
        * year_multiplier
        * count_multiplier
    )

    avg_severity_score = (
        sum(severity_scores) / len(severity_scores) if severity_scores else 1
    )

    if avg_severity_score >= 2.5:
        overall_severity = "High"
    elif avg_severity_score >= 1.6:
        overall_severity = "Moderate"
    else:
        overall_severity = "Low"

    return {
        "min": round(final_min),
        "max": round(final_max),
        "currency": "USD",
        "severity": overall_severity,
        "multipliers": {
            "brand": brand_multiplier,
            "vehicle_type": vehicle_type_multiplier,
            "year": year_multiplier,
            "damage_count": count_multiplier,
        },
        "summary": (
            f"The system detected {len(detections)} possible damage area(s). "
            f"The estimated severity is {overall_severity}. "
            f"The estimated repair range is ${round(final_min):,} - ${round(final_max):,} USD."
        ),
        "disclaimer": (
            "This is an AI-generated preliminary estimate and not an official repair quote. "
            "Final repair cost may vary by location, labor rate, parts availability, paint matching, "
            "hidden damage, calibration needs, and repair shop pricing."
        ),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Car Damage API is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    make: str = Form(""),
    model_name: str = Form(""),
    year: str = Form(""),
    vehicle_type: str = Form("sedan"),
):
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

    try:
        parsed_year = int(year)
    except (TypeError, ValueError):
        parsed_year = None

    repair_estimate = calculate_repair_estimate(
        detections=detections,
        make=make,
        vehicle_type=vehicle_type,
        year=parsed_year,
    )

    return {
        "vehicle": {
            "make": make,
            "model": model_name,
            "year": parsed_year,
            "vehicle_type": vehicle_type,
        },
        "detections": detections,
        "annotated_image_url": f"/outputs/{output_filename}",
        "repair_estimate": repair_estimate,
    }