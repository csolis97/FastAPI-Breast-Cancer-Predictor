from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

app = FastAPI()

model = None
scaler = None

@app.on_event("startup")
def load_model_and_scaler():
    global model, scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise RuntimeError("Modelos no encontrados al iniciar el servidor.")

# --- Handler para errores de validación ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    mensajes = []
    for error in exc.errors():
        # Usamos el msg del ValueError lanzado en el validador
        if error["type"] == "value_error":
            mensajes.append(error["msg"])
        else:
            mensajes.append(error)
    
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validación fallida. Revisa los campos enviados.",
            "detail": mensajes,
            "body": exc.body,
        },
    )

# --- Modelo con validaciones ---
class InputData(BaseModel):
    mean_radius: float = Field(..., example=14.127292)
    mean_texture: float = Field(..., example=19.289649)
    mean_perimeter: float = Field(..., example=91.969033)
    mean_area: float = Field(..., example=654.889104)
    mean_smoothness: float = Field(..., example=0.096360)
    mean_compactness: float = Field(..., example=0.104341)
    mean_concavity: float = Field(..., example=0.088799)
    mean_concave_points: float = Field(..., example=0.048919)
    mean_symmetry: float = Field(..., example=0.181162)
    mean_fractal_dimension: float = Field(..., example=0.062798)
    radius_error: float = Field(..., example=0.405172)
    texture_error: float = Field(..., example=1.216853)
    perimeter_error: float = Field(..., example=2.866059)
    area_error: float = Field(..., example=40.337079)
    smoothness_error: float = Field(..., example=0.007041)
    compactness_error: float = Field(..., example=0.025478)
    concavity_error: float = Field(..., example=0.031894)
    concave_points_error: float = Field(..., example=0.011796)
    symmetry_error: float = Field(..., example=0.020542)
    fractal_dimension_error: float = Field(..., example=0.003795)
    worst_radius: float = Field(..., example=16.269190)
    worst_texture: float = Field(..., example=25.677223)
    worst_perimeter: float = Field(..., example=107.261213)
    worst_area: float = Field(..., example=880.583128)
    worst_smoothness: float = Field(..., example=0.132369)
    worst_compactness: float = Field(..., example=0.254265)
    worst_concavity: float = Field(..., example=0.272188)
    worst_concave_points: float = Field(..., example=0.114606)
    worst_symmetry: float = Field(..., example=0.290076)
    worst_fractal_dimension: float = Field(..., example=0.083946)
    
    @validator('*', pre=True)
    def no_negativos(cls, v):
        if v is None:
            return v
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError('Los valores no pueden ser negativos')
        if v > 9999:
            raise ValueError('Valor atípico detectado: 9999 no es válido')
        return v

@app.get("/")
def read_root():
    return {"message": "API funcionando. Usar POST /predict para hacer predicciones."}

@app.post("/predict")
def predict(data: InputData):
    if scaler is None or model is None:
        raise HTTPException(status_code=500, detail="Modelo o scaler no están cargados correctamente.")
    try:
        input_data = np.array([[
            data.mean_radius,
            data.mean_texture,
            data.mean_perimeter,
            data.mean_area,
            data.mean_smoothness,
            data.mean_compactness,
            data.mean_concavity,
            data.mean_concave_points,
            data.mean_symmetry,
            data.mean_fractal_dimension,
            data.radius_error,
            data.texture_error,
            data.perimeter_error,
            data.area_error,
            data.smoothness_error,
            data.compactness_error,
            data.concavity_error,
            data.concave_points_error,
            data.symmetry_error,
            data.fractal_dimension_error,
            data.worst_radius,
            data.worst_texture,
            data.worst_perimeter,
            data.worst_area,
            data.worst_smoothness,
            data.worst_compactness,
            data.worst_concavity,
            data.worst_concave_points,
            data.worst_symmetry,
            data.worst_fractal_dimension
        ]])

        features_scaled = scaler.transform(input_data)

        prediction = model.predict(features_scaled)
        pred_proba = model.predict_proba(features_scaled)[:, 1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(pred_proba)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
