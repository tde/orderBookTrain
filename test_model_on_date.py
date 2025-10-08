#!/usr/bin/env python3
"""
Скрипт для тестирования модели на заданной дате

Использование:
    python test_model_on_date.py --model model_2025-09-22.pt --test-date 2025-09-30
    
Или в Jupyter:
    %run test_model_on_date.py --model model_2025-09-22.pt --test-date 2025-09-30
"""

import sys
import os
from pathlib import Path
import argparse

# Добавляем src в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from config import Config
from load_data import load_data
from tools import build_barrier_labels
from model import create_model
from dataset import LazyWindowDataset
from torch.utils.data import DataLoader


def evaluate_model(model, data_loader, device, num_classes=3):
    """Оценка модели на данных"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            
            num_batches += 1
    
    # Объединяем все батчи
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    # Вычисляем метрики
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_true
    }


def load_test_data(test_date, config):
    """Загрузка тестовых данных для заданной даты"""
    print(f"\n📥 Загрузка данных для {test_date}...")
    
    features_file = config.data.get_features_file(test_date)
    prices_file = config.data.get_prices_file(test_date)
    
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Файл не найден: {features_file}")
    if not os.path.exists(prices_file):
        raise FileNotFoundError(f"Файл не найден: {prices_file}")
    
    X, ms, mid = load_data(features_file, prices_file)
    
    y_all = build_barrier_labels(
        ms, mid,
        tick_size=config.model.tick_size,
        theta_ticks=config.model.theta_ticks,
        horizon_sec=config.model.horizon_sec
    )
    
    valid = (y_all != -1)
    Xv, yv = X[valid], y_all[valid]
    
    test_ds = LazyWindowDataset(Xv, yv, config.model.window_length)
    test_dl = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    
    y_win = yv[config.model.window_length-1:]
    
    print(f"  Всего срезов: {len(X)}")
    print(f"  Валидных: {len(Xv)}")
    print(f"  Окон: {len(test_ds)}")
    print(f"  Распределение классов: {np.unique(y_win, return_counts=True)}")
    
    return test_dl, y_win


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели на заданной дате')
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу модели (например, model_2025-09-22.pt)')
    parser.add_argument('--test-date', type=str, required=True, help='Дата для тестирования (например, 2025-09-30)')
    parser.add_argument('--data-folder', type=str, default='./data/npy', help='Папка с данными')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*70)
    print(f"\n📦 Файл модели: {args.model}")
    print(f"📅 Дата теста: {args.test_date}")
    print(f"💻 Устройство: {device}")
    
    # Проверяем существование файла модели
    if not os.path.exists(args.model):
        print(f"\n❌ ОШИБКА: Файл модели не найден: {args.model}")
        print("\n📂 Доступные файлы моделей:")
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if model_files:
            for f in sorted(model_files):
                print(f"   - {f}")
        else:
            print("   (нет файлов .pt в текущей директории)")
        return
    
    # Создаем конфигурацию
    config = Config.default()
    config.data.data_folder = args.data_folder
    config.model.tick_size = 1.0
    config.model.theta_ticks = 5
    config.model.horizon_sec = 2.0
    config.model.window_length = 240
    config.training.batch_size = 512
    
    # Загружаем модель
    print(f"\n⚙️  Создание архитектуры модели...")
    model, _ = create_model(config.model)
    
    print(f"📥 Загрузка весов из {args.model}...")
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Модель загружена!")
    print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Загружаем тестовые данные
    try:
        test_dl, y_true = load_test_data(args.test_date, config)
    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА: {e}")
        return
    
    # Выполняем тестирование
    print(f"\n🔄 Выполнение inference...")
    results = evaluate_model(model, test_dl, device, num_classes=config.model.num_classes)
    
    # Выводим результаты
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)
    
    print(f"\n🎯 МЕТРИКИ:")
    print(f"  Macro F1:     {results['f1_macro']:.4f}")
    print(f"  F1 ↓ (down):  {results['f1_per_class'][0]:.4f}")
    print(f"  F1 ○ (flat):  {results['f1_per_class'][1]:.4f}")
    print(f"  F1 ↑ (up):    {results['f1_per_class'][2]:.4f}")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"  (rows=true, cols=pred)")
    cm = results['confusion_matrix']
    print(f"\n         Down    Flat      Up")
    print(f"Down   [{cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}]")
    print(f"Flat   [{cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}]")
    print(f"Up     [{cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}]")
    
    # Точность по классам
    print(f"\n📐 ТОЧНОСТЬ ПО КЛАССАМ:")
    for i, class_name in enumerate(['Down', 'Flat', 'Up']):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        print(f"  {class_name:5s}: {correct:6d}/{total:6d} = {accuracy:.3f}")
    
    # Общая точность
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_accuracy = total_correct / total_samples
    print(f"\n  Overall: {total_correct:6d}/{total_samples:6d} = {overall_accuracy:.3f}")
    
    print("\n" + "="*70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

