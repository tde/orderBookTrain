import os
import numpy as np
import pandas as pd
from typing import Tuple


def load_data(features_file: str, prices_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загрузка данных из файлов
    
    Args:
        features_file: Путь к файлу с признаками (.npy)
        prices_file: Путь к файлу с ценами (.csv)
        
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