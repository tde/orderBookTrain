import os
import numpy as np
import pandas as pd
from typing import Tuple


def load_data(features_file: str, prices_file: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загрузка данных из файлов
    
    Args:
        features_file: Путь к файлу с признаками (.npy)
        prices_file: Путь к файлу с ценами (.csv)
        normalize: Применять ли нормализацию к признакам (по умолчанию True)
        
    Returns:
        Кортеж (X, ms, mid) где:
        - X: массив признаков
        - ms: временные метки в миллисекундах
        - mid: средние цены
    """
    print(f"Загрузка признаков: {features_file}")
    print(f"Загрузка цен: {prices_file}")
    
    # Загрузка фичей и меток
    X = np.load(features_file)
    df_p = pd.read_csv(prices_file)
    
    assert {"ms", "mid"}.issubset(df_p.columns), "Нужны колонки ms и mid в prices.csv"
    
    print("Форма массива:", X.shape)
    print("Тип данных:", X.dtype)
    
    # Нормализация данных (важно для стабильного обучения!)
    if normalize:
        # Z-score нормализация по каждому признаку
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        # Избегаем деления на 0
        std = np.where(std < 1e-8, 1.0, std)
        X = (X - mean) / std
        print("✓ Данные нормализованы (z-score)")
        print(f"  Диапазон значений: [{X.min():.2f}, {X.max():.2f}]")
    
    ms = df_p["ms"].values.astype(np.int64)
    mid = df_p["mid"].values.astype(np.float64)
    
    assert len(X) == len(ms) == len(mid), f"Несовпадение длин: X={len(X)} ms={len(ms)} mid={len(mid)}"
    
    return X, ms, mid


def main():
    """Основная функция для запуска из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_date", required=True, help="Дата для обучения")
    parser.add_argument("--data_folder", default="data", help="Папка с данными")
    parser.add_argument("--instrument", default="Si-12.25", help="Инструмент")
    
    args = parser.parse_args()
    
    # Пути к файлам
    features_file = f"{args.data_folder}/{args.instrument}_{args.train_date}_features.npy"
    prices_file = f"{args.data_folder}/{args.instrument}_{args.train_date}_prices.csv"
    
    # Загрузка данных
    X, ms, mid = load_data(features_file, prices_file)
    
    print("Данные успешно загружены!")


if __name__ == "__main__":
    main()