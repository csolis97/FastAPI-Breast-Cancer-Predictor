# FastAPI-Breast-Cancer-Predictor

Este proyecto implementa una **API en FastAPI** que carga un modelo previamente entrenado y expone un endpoint `/predict` para realizar predicciones a partir de datos enviados en formato JSON.

---
## Estructura del repositorio

```
FastAPI-Breast-Cancer-Predictor/
│── notebooks        # Fuente del modelo y simulador cliente
│── models           # Modelo entrenado y scaler
│── main.py          # Archivo principal de la API
│── requirements.txt # Dependencias
│── README.md        # Documentación completa
```
---
## Instalación y configuración local

1. **Clonar este repositorio**:

```bash
git clone https://github.com/csolis97/FastAPI-Breast-Cancer-Predictor
cd FastAPI-Breast-Cancer-Predictor
```
2. **Abre CMD y entra a la carpeta donde clonaste el  proyecto (ajusta tu ruta)**:

```bash
cd "C:\Users\xx\Desktop\FastAPI-Breast-Cancer-Predictor"
```
3. **Crear un entorno virtual** (opcional pero recomendado):

```bash
python -m venv venv
```
4. **Activar el entorno virtual**:

```bash
venv\Scripts\activate.bat
```
5. **Instalar las dependencias**:

```bash
pip install -r requirements.txt
```
5. **Iniciar el servidor local**:

```bash
uvicorn main:app --reload
```
El servicio quedará disponible en:\
`http://127.0.0.1:8000`

---

## Uso del endpoint `/predict`

### **Método:**

`POST`

### **JSON de entrada esperado**:
Cada campo corresponde a un atributo del tumor. Todos son de tipo float positivo.  

```json
{
    "mean_radius": 14.127292,
    "mean_texture": 19.289649,
    "mean_perimeter": 91.969033,
    "mean_area": 654.889104,
    "mean_smoothness": 0.096360,
    "mean_compactness": 0.104341,
    "mean_concavity": 0.088799,
    "mean_concave_points": 0.048919,
    "mean_symmetry": 0.181162,
    "mean_fractal_dimension": 0.062798,
    "radius_error": 0.405172,
    "texture_error": 1.216853,
    "perimeter_error": 2.866059,
    "area_error": 40.337079,
    "smoothness_error": 0.007041,
    "compactness_error": 0.025478,
    "concavity_error": 0.031894,
    "concave_points_error": 0.011796,
    "symmetry_error": 0.020542,
    "fractal_dimension_error": 0.003795,
    "worst_radius": 16.269190,
    "worst_texture": 25.677223,
    "worst_perimeter": 107.261213,
    "worst_area": 880.583128,
    "worst_smoothness": 0.132369,
    "worst_compactness": 0.254265,
    "worst_concavity": 0.32,
    "worst_concave_points": 0.114606,
    "worst_symmetry": 0.290076,
    "worst_fractal_dimension": 0.083946
}
```

### **Ejemplo de respuesta**:

```json
{
    "'prediction": "1",
    "probabilidad": 0.5332673580926112
}
```
###  Pruebas desde un cliente externo

El repositorio incluye un notebook (`notebooks/01_client.ipynb`) que
realiza al menos **3 peticiones distintas** a la API, mostrando los
datos enviados y los resultados obtenidos.

También puedes crear tus propias pruebas con `requests` en Python o
Postman.

### API desplegada en Render
🌐 <https://fastapi-breast-cancer-predictor-uyv6.onrender.com/docs>

---
## Autor
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Cristhian%20Solís-blue?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/csolism97/)
