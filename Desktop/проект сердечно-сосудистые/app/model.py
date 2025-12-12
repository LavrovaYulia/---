import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HeartAttackModel:
    def __init__(self, model_path: str):
        """
        Инициализация модели
        
        Args:
            model_path: Путь к сохраненной модели .pkl
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Загрузка модели из файла"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            
            # Получение имен признаков
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            elif hasattr(self.model.named_steps.get('classifier', {}), 'feature_importances_'):
                # Для пайплайнов
                self.feature_names = list(self.model.named_steps['preprocessor'].get_feature_names_out())
            else:
                self.feature_names = []
            
            logger.info(f"Модель загружена успешно. Признаков: {len(self.feature_names) if self.feature_names else 'неизвестно'}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных
        
        Args:
            df: Входной DataFrame
        
        Returns:
            Обработанный DataFrame
        """
        # Создаем копию
        df_processed = df.copy()
        
        # Приведение названий столбцов к нижнему регистру
        df_processed.columns = df_processed.columns.str.lower()
        
        # Удаление ненужных столбцов
        drop_cols = ['unnamed:_0', 'id', 'income']
        df_processed = df_processed.drop(columns=[col for col in drop_cols if col in df_processed.columns], errors='ignore')
        
        # Обработка gender
        if 'gender' in df_processed.columns:
            if df_processed['gender'].dtype == 'object':
                gender_map = {'male': 0, 'female': 1, '1.0': 0, '0.0': 1, '1': 0, '0': 1}
                df_processed['gender'] = df_processed['gender'].astype(str).str.lower().map(gender_map).fillna(0).astype(int)
        
        # Бинарные признаки
        binary_cols = ['diabetes', 'family_history', 'smoking', 'obesity',
                      'alcohol_consumption', 'previous_heart_problems', 'medication_use']
        
        for col in binary_cols:
            if col in df_processed.columns:
                if df_processed[col].dtype == 'float64':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median()).round().astype(int)
        
        # Заполнение пропусков
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        return df_processed
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Пакетное предсказание
        
        Args:
            df: DataFrame с данными пациентов
        
        Returns:
            DataFrame с ID и предсказаниями
        """
        # Предобработка
        df_processed = self.preprocess_data(df)
        
        # Предсказание
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(df_processed)[:, 1]
        else:
            predictions = self.model.predict(df_processed)
        
        # Создание результата
        result_df = pd.DataFrame({
            'id': df['id'].values if 'id' in df.columns else range(len(df)),
            'prediction': predictions
        })
        
        return result_df
    
    def predict_single(self, df: pd.DataFrame) -> float:
        """
        Предсказание для одного пациента
        
        Args:
            df: DataFrame с данными одного пациента
        
        Returns:
            Вероятность сердечного приступа
        """
        result_df = self.predict_batch(df)
        return result_df['prediction'].iloc[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        info = {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "features_count": len(self.feature_names) if self.feature_names else "unknown",
            "model_loaded": self.model is not None
        }
        
        if self.model:
            # Информация о пайплайне
            if hasattr(self.model, 'named_steps'):
                steps = list(self.model.named_steps.keys())
                info["pipeline_steps"] = steps
            
            # Информация о классификаторе
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                info["classifier_type"] = type(self.model.named_steps.get('classifier', self.model)).__name__
        
        return info