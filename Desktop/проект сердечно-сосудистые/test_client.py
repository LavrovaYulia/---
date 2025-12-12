import requests
import pandas as pd
import json
import time
import os

class HeartAttackAPIClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Проверка работоспособности API"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        """Получение информации о модели"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_csv(self, csv_path):
        """
        Отправка CSV файла для предсказаний
        
        Args:
            csv_path: Путь к CSV файлу
        
        Returns:
            Результаты предсказаний
        """
        try:
            with open(csv_path, 'rb') as f:
                files = {'file': (os.path.basename(csv_path), f, 'text/csv')}
                response = requests.post(
                    f"{self.base_url}/predict/csv",
                    files=files
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Скачивание файла с предсказаниями
                if 'download_url' in result:
                    download_url = f"{self.base_url}{result['download_url']}"
                    download_response = requests.get(download_url)
                    
                    if download_response.status_code == 200:
                        # Сохранение предсказаний
                        output_path = "api_predictions.csv"
                        with open(output_path, 'wb') as f:
                            f.write(download_response.content)
                        print(f"Предсказания сохранены в {output_path}")
                
                return result
            else:
                return {"error": f"Status code: {response.status_code}", "detail": response.text}
                
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, patient_data):
        """
        Предсказание для одного пациента
        
        Args:
            patient_data: Словарь с данными пациента
        
        Returns:
            Результат предсказания
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict/single",
                json=patient_data
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_all_endpoints(self):
        """Тестирование всех эндпоинтов"""
        print("=" * 50)
        print("ТЕСТИРОВАНИЕ API")
        print("=" * 50)
        
        # 1. Проверка здоровья
        print("\n1. Проверка здоровья API:")
        health = self.health_check()
        print(json.dumps(health, indent=2, ensure_ascii=False))
        
        # 2. Информация о модели
        print("\n2. Информация о модели:")
        model_info = self.get_model_info()
        print(json.dumps(model_info, indent=2, ensure_ascii=False))
        
        # 3. Тестовые данные для одного пациента
        print("\n3. Тест предсказания для одного пациента:")
        test_patient = {
            "id": "test_001",
            "age": 55,
            "cholesterol": 240,
            "heart_rate": 72,
            "diabetes": 1,
            "family_history": 1,
            "smoking": 1,
            "obesity": 0,
            "alcohol_consumption": 1,
            "exercise_hours_per_week": 3,
            "diet": 2,
            "previous_heart_problems": 0,
            "medication_use": 1,
            "stress_level": 3,
            "sedentary_hours_per_day": 8,
            "bmi": 28.5,
            "triglycerides": 150,
            "physical_activity_days_per_week": 3,
            "sleep_hours_per_day": 7,
            "blood_sugar_level": 120,
            "ck_-_mb": 25,
            "troponin": 0.01,
            "gender": 1,
            "systolic_blood_pressure": 140,
            "diastolic_blood_pressure": 90
        }
        
        single_pred = self.predict_single(test_patient)
        print(json.dumps(single_pred, indent=2, ensure_ascii=False))
        
        return health, model_info, single_pred

def main():
    # Инициализация клиента
    client = HeartAttackAPIClient()
    
    # Тестирование всех эндпоинтов
    client.test_all_endpoints()
    
    # Проверка существования тестового файла
    test_csv_path = "data/heart_test.csv"
    if os.path.exists(test_csv_path):
        print(f"\n4. Тестирование загрузки CSV файла: {test_csv_path}")
        
        # Создание мини-файла для теста (первые 5 строк)
        test_df = pd.read_csv(test_csv_path)
        mini_test_path = "test_sample.csv"
        test_df.head(5).to_csv(mini_test_path, index=False)
        
        result = client.predict_from_csv(mini_test_path)
        print("Результат предсказаний CSV:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Удаление временного файла
        os.remove(mini_test_path)
    else:
        print(f"\nТестовый файл не найден: {test_csv_path}")
        print("Создайте папку data и поместите туда heart_test.csv")

if __name__ == "__main__":
    main()