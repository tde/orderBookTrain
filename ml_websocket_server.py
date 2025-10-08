#!/usr/bin/env python3
"""
WebSocket сервер для real-time inference модели DeepLOB
Принимает срезы стакана, накапливает окно 240 срезов, делает прогноз
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import torch
import websockets

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from model import create_model
from config import Config

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Путь к файлу модели
MODEL_PATH = "averaged_model_days_1_to_4.pt"  # Измените на нужный файл

# Параметры сервера
HOST = "localhost"
PORT = 8765

# Размер окна (должен совпадать с window_length при обучении)
WINDOW_SIZE = 240

# Количество признаков (должно совпадать с input_features)
NUM_FEATURES = 211

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================

model = None
device = None
config = None

# Словарь для хранения окон каждого подключения
# client_id -> deque of features
client_windows = {}

# Статистика
stats = {
    "total_predictions": 0,
    "clients_connected": 0,
    "start_time": None
}

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model(model_path: str):
    """Загрузить модель из файла"""
    global model, device, config
    
    print(f"\n{'='*70}")
    print("🚀 ЗАГРУЗКА ML МОДЕЛИ")
    print(f"{'='*70}\n")
    
    # Проверка наличия файла
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ Файл модели не найден: {model_path}")
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Устройство: {device}")
    
    # Создаем конфигурацию
    config = Config.default()
    
    # Создаем модель
    print(f"🔧 Создание архитектуры модели...")
    model, _ = create_model(config.model)
    
    # Загружаем веса
    print(f"📥 Загрузка весов из {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Переводим в режим inference
    model = model.to(device)
    model.eval()
    
    print(f"✅ Модель загружена успешно!")
    print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Input shape: (batch, {NUM_FEATURES}, {WINDOW_SIZE})")
    
    # Warmup - прогреваем модель
    print(f"\n🔥 Warmup модели...")
    with torch.no_grad():
        dummy_input = torch.randn(1, NUM_FEATURES, WINDOW_SIZE).to(device)
        _ = model(dummy_input)
    print(f"✅ Warmup завершен")
    
    print(f"\n{'='*70}\n")
    return model

# ============================================================
# INFERENCE
# ============================================================

def make_prediction(features_window: np.ndarray):
    """
    Сделать предсказание на основе окна признаков
    
    Args:
        features_window: numpy array shape (240, 211)
        
    Returns:
        dict with prediction, probabilities, confidence
    """
    global model, device, stats
    
    try:
        # Конвертируем в tensor
        # Входной формат модели: (batch, features, time) = (1, 211, 240)
        # У нас: (240, 211) -> transpose -> (211, 240) -> add batch dim -> (1, 211, 240)
        x = torch.from_numpy(features_window.T).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            # Получаем предсказание
            pred_idx = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_idx].item()
            
            # Маппинг индекса в класс
            class_names = ["down", "flat", "up"]
            prediction = class_names[pred_idx]
            
            # Вероятности всех классов
            probabilities = {
                "down": float(probs[0, 0].item()),
                "flat": float(probs[0, 1].item()),
                "up": float(probs[0, 2].item())
            }
        
        # Обновляем статистику
        stats["total_predictions"] += 1
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }
        
    except Exception as e:
        print(f"❌ Ошибка при inference: {e}")
        raise

# ============================================================
# WEBSOCKET HANDLERS
# ============================================================

async def handle_client(websocket, path):
    """Обработка подключения клиента"""
    client_id = id(websocket)
    client_address = websocket.remote_address
    
    print(f"✅ Новое подключение: {client_address} (ID: {client_id})")
    stats["clients_connected"] += 1
    
    # Создаем окно для этого клиента (фиксированная очередь)
    client_windows[client_id] = deque(maxlen=WINDOW_SIZE)
    
    try:
        async for message in websocket:
            try:
                # Парсим входящее сообщение
                data = json.loads(message)
                
                # Валидация данных
                if "features" not in data:
                    await websocket.send(json.dumps({
                        "error": "Missing 'features' field"
                    }))
                    continue
                
                features = np.array(data["features"], dtype=np.float32)
                
                # Проверка размерности
                if features.shape[0] != NUM_FEATURES:
                    await websocket.send(json.dumps({
                        "error": f"Expected {NUM_FEATURES} features, got {features.shape[0]}"
                    }))
                    continue
                
                # Добавляем срез в окно (автоматически удаляется самый старый при переполнении)
                client_windows[client_id].append(features)
                
                current_window_size = len(client_windows[client_id])
                
                # Если окно заполнено - делаем предсказание
                if current_window_size == WINDOW_SIZE:
                    # Конвертируем deque в numpy array (240, 211)
                    features_window = np.array(client_windows[client_id])
                    
                    # Замеряем время inference
                    start_time = asyncio.get_event_loop().time()
                    
                    # Делаем предсказание
                    result = make_prediction(features_window)
                    
                    inference_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    
                    # Формируем ответ
                    response = {
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                        "prediction": result["prediction"],
                        "confidence": round(result["confidence"], 4),
                        "probabilities": {
                            k: round(v, 4) for k, v in result["probabilities"].items()
                        },
                        "window_size": current_window_size,
                        "inference_time_ms": round(inference_time_ms, 2)
                    }
                    
                    # Добавляем request_id если был в запросе
                    if "request_id" in data:
                        response["request_id"] = data["request_id"]
                    
                    # Отправляем результат
                    await websocket.send(json.dumps(response))
                    
                    # Логируем
                    print(f"📊 [{client_id}] Prediction: {result['prediction']} "
                          f"(conf: {result['confidence']:.3f}, "
                          f"latency: {inference_time_ms:.1f}ms)")
                    
                else:
                    # Окно еще не заполнено
                    await websocket.send(json.dumps({
                        "status": "buffering",
                        "window_size": current_window_size,
                        "required": WINDOW_SIZE,
                        "message": f"Buffering data... {current_window_size}/{WINDOW_SIZE}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                print(f"❌ Ошибка обработки сообщения: {e}")
                await websocket.send(json.dumps({
                    "error": str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"❌ Соединение закрыто: {client_address} (ID: {client_id})")
    finally:
        # Очистка при отключении
        if client_id in client_windows:
            del client_windows[client_id]
        stats["clients_connected"] -= 1
        print(f"🔌 Клиент отключен: {client_address} (ID: {client_id})")

# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================

async def main():
    """Главная функция сервера"""
    global stats
    
    print("\n" + "="*70)
    print("🤖 ML WEBSOCKET SERVER")
    print("="*70)
    
    # Загружаем модель
    try:
        load_model(MODEL_PATH)
    except Exception as e:
        print(f"\n❌ Ошибка загрузки модели: {e}")
        print("Убедитесь что файл модели существует и путь указан правильно.")
        return
    
    # Запускаем WebSocket сервер
    stats["start_time"] = datetime.now()
    
    print(f"🌐 Запуск WebSocket сервера...")
    print(f"   URL: ws://{HOST}:{PORT}")
    print(f"   Ожидаемый формат: {NUM_FEATURES} признаков")
    print(f"   Размер окна: {WINDOW_SIZE} срезов")
    print(f"\n{'='*70}")
    print("✅ Сервер готов к приему подключений!")
    print("   Нажмите Ctrl+C для остановки")
    print(f"{'='*70}\n")
    
    # Запускаем сервер
    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()  # run forever

# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("🛑 Остановка сервера...")
        print(f"{'='*70}")
        
        # Выводим статистику
        uptime = (datetime.now() - stats["start_time"]).total_seconds() if stats["start_time"] else 0
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   Время работы: {uptime:.1f} секунд")
        print(f"   Всего предсказаний: {stats['total_predictions']}")
        print(f"   Активных подключений: {stats['clients_connected']}")
        
        print(f"\n✅ Сервер остановлен\n")

