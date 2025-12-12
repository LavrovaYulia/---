from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import List, Optional
import io
from fastapi.openapi.docs import get_redoc_html
from .model import HeartAttackModel
from .schemas import PredictionResponse, BatchPredictionResponse

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Heart Attack Prediction API",
    description="API для предсказания риска сердечного приступа",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    )

# Загрузка модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
try:
    model = HeartAttackModel(MODEL_PATH)
    logger.info(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

@app.get("/")
async def root():
    """Корневая страница API"""
    return {
        "message": "Heart Attack Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Документация Swagger",
            "/predict/csv": "Загрузка CSV файла для предсказаний",
            "/health": "Проверка работоспособности API",
            "/model/info": "Информация о модели"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка работоспособности API и модели"""
    status = {
        "api": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else None
    }
    if model:
        status.update(model.get_model_info())
    
    return JSONResponse(content=status)

@app.get("/model/info")
async def model_info():
    """Получение информации о модели"""
    if not model:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return model.get_model_info()

@app.post("/predict/csv", response_model=BatchPredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Получение предсказаний из CSV файла
    
    - **file**: CSV файл с данными для предсказания
    - **returns**: JSON с предсказаниями и возможностью скачать CSV
    """
    if not model:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Загружен файл: {file.filename}, строк: {len(df)}")
        
        # Получение предсказаний
        predictions_df = model.predict_batch(df)
        
        # Конвертация в JSON
        predictions_json = predictions_df.to_dict(orient='records')
        
        # Сохранение временного файла с предсказаниями
        output_path = "temp_predictions.csv"
        predictions_df.to_csv(output_path, index=False)
        
        return BatchPredictionResponse(
            message=f"Обработано {len(predictions_df)} записей",
            predictions=predictions_json,
            download_url=f"/download/{output_path}",
            statistics={
                "mean": float(predictions_df['prediction'].mean()),
                "min": float(predictions_df['prediction'].min()),
                "max": float(predictions_df['prediction'].max()),
                "std": float(predictions_df['prediction'].std()),
                "risk_count": int((predictions_df['prediction'] > 0.5).sum()),
                "no_risk_count": int((predictions_df['prediction'] <= 0.5).sum())
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Скачивание файла с предсказаниями
    
    - **filename**: имя файла для скачивания
    """
    if filename != "temp_predictions.csv":
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        filename, 
        media_type='text/csv',
        filename="heart_attack_predictions.csv"
    )

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(data: dict):
    """
    Получение предсказания для одного пациента
    
    - **data**: JSON с данными пациента
    """
    if not model:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Конвертация в DataFrame
        df = pd.DataFrame([data])
        
        # Получение предсказания
        prediction = model.predict_single(df)
        
        return PredictionResponse(
            patient_id=data.get('id', 'unknown'),
            prediction=float(prediction),
            risk_level="Высокий" if prediction > 0.5 else "Низкий",
            confidence=float(abs(prediction - 0.5) * 2)  # Уверенность в предсказании
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки данных: {str(e)}")

# Middleware для логирования
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Запрос: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Ответ: {response.status_code}")
    return response