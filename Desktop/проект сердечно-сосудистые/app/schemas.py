from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionResponse(BaseModel):
    """Схема ответа для одного предсказания"""
    patient_id: str
    prediction: float
    risk_level: str
    confidence: float

class BatchPredictionResponse(BaseModel):
    """Схема ответа для пакетного предсказания"""
    message: str
    predictions: List[Dict[str, Any]]
    download_url: str
    statistics: Dict[str, Any]

class HealthResponse(BaseModel):
    """Схема ответа для проверки здоровья"""
    class Config:
        protected_namespaces = ()
    
    api: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    features_count: Optional[int] = None

class ModelInfoResponse(BaseModel):
    """Схема ответа с информацией о модели"""
    class Config:
        protected_namespaces = ()
    
    model_path: str
    model_type: str
    features_count: int
    model_loaded: bool
    pipeline_steps: Optional[List[str]] = None
    classifier_type: Optional[str] = None