from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = joblib.load("flood_model.pkl")
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
    # При заходе покажем форму без результата
    return templates.TemplateResponse("form.html", {"request": request, "features": features_list, "result": None})

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
    prediction = model.predict(scaled)[0]

    return templates.TemplateResponse(
        "form.html",
        {"request": request, "features": features_list, "result": int(prediction)}
    )
