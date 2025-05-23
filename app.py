from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загрузка моделей
classifier = joblib.load("flood_model.pkl")
regressor = joblib.load("flood_regressor.pkl")
scaler = joblib.load("flood_scaler.pkl")

features_list = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability",
    "Landslides", "Watersheds", "DeterioratingInfrastructure", "PopulationScore",
    "WetlandLoss", "InadequatePlanning", "PoliticalFactors"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "features": features_list,
        "result_classification": None,
        "result_regression": None
    })

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    MonsoonIntensity: float = Form(...),
    TopographyDrainage: float = Form(...),
    RiverManagement: float = Form(...),
    Deforestation: float = Form(...),
    Urbanization: float = Form(...),
    ClimateChange: float = Form(...),
    DamsQuality: float = Form(...),
    Siltation: float = Form(...),
    AgriculturalPractices: float = Form(...),
    Encroachments: float = Form(...),
    IneffectiveDisasterPreparedness: float = Form(...),
    DrainageSystems: float = Form(...),
    CoastalVulnerability: float = Form(...),
    Landslides: float = Form(...),
    Watersheds: float = Form(...),
    DeterioratingInfrastructure: float = Form(...),
    PopulationScore: float = Form(...),
    WetlandLoss: float = Form(...),
    InadequatePlanning: float = Form(...),
    PoliticalFactors: float = Form(...)
):
    features = [
        MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation,
        Urbanization, ClimateChange, DamsQuality, Siltation, AgriculturalPractices,
        Encroachments, IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability,
        Landslides, Watersheds, DeterioratingInfrastructure, PopulationScore,
        WetlandLoss, InadequatePlanning, PoliticalFactors
    ]
    scaled = scaler.transform([features])

    classification_result = classifier.predict(scaled)[0]
    regression_result = regressor.predict(scaled)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "features": features_list,
        "result_classification": int(classification_result),
        "result_regression": round(float(regression_result), 2)
    })
