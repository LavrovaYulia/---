import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_csv_structure(df: pd.DataFrame) -> bool:
    """
    Валидация структуры CSV файла
    
    Args:
        df: DataFrame для проверки
    
    Returns:
        True если структура корректна
    """
    required_columns = ['age', 'cholesterol', 'heart_rate']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Отсутствует обязательный столбец: {col}")
            return False
    
    if df.empty:
        logger.error("CSV файл пуст")
        return False
    
    return True

def save_predictions(predictions_df: pd.DataFrame, filename: str = "predictions.csv"):
    """
    Сохранение предсказаний в CSV файл
    
    Args:
        predictions_df: DataFrame с предсказаниями
        filename: Имя файла для сохранения
    """
    predictions_df.to_csv(filename, index=False)
    logger.info(f"Предсказания сохранены в {filename}")
    return filename

def calculate_statistics(predictions: np.ndarray) -> Dict[str, float]:
    """
    Расчет статистики по предсказаниям
    
    Args:
        predictions: Массив предсказаний
    
    Returns:
        Словарь со статистикой
    """
    return {
        "mean": float(np.mean(predictions)),
        "median": float(np.median(predictions)),
        "std": float(np.std(predictions)),
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "q25": float(np.percentile(predictions, 25)),
        "q75": float(np.percentile(predictions, 75))
    }