#!/usr/bin/env python3
"""
Основной скрипт для обучения модели DeepLOB
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from load_data import load_data
from tools import build_barrier_labels, make_windows
from model import create_model, create_optimizer_and_scheduler
from dataset import NpWindowDataset, BalancedBatchSampler, BalancedBatchBatchSampler
from trainer import create_trainer
from torch.utils.data import DataLoader


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Обучение модели DeepLOB")
    
    # Параметры данных
    parser.add_argument("--train_dates", type=str, default="2025-09-22",
                       help="Список дат для обучения (через запятую)")
    parser.add_argument("--val_date", type=str, default="2025-09-23",
                       help="Дата для валидации")
    parser.add_argument("--test_date", type=str, default="2025-09-24",
                       help="Дата для тестирования")
    parser.add_argument("--data_folder", type=str, default="data",
                       help="Папка с данными")
    parser.add_argument("--instrument", type=str, default="Si-12.25",
                       help="Инструмент")
    
    # Параметры модели
    parser.add_argument("--tick_size", type=float, default=1.0,
                       help="Размер тика инструмента")
    parser.add_argument("--theta_ticks", type=int, default=5,
                       help="Порог в тиках")
    parser.add_argument("--horizon_sec", type=float, default=2.0,
                       help="Горизонт в секундах")
    parser.add_argument("--window_length", type=int, default=240,
                       help="Длина окна")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Размер скрытого слоя")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Коэффициент dropout")
    parser.add_argument("--use_cost_sensitive", action="store_true", default=True,
                       help="Использовать стоимостно-чувствительный лосс")
    parser.add_argument("--use_focal_loss", action="store_true", default=False,
                       help="Использовать Focal Loss вместо стоимостно-чувствительного")
    
    # Параметры обучения
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Размер батча")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Скорость обучения")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Количество эпох")
    parser.add_argument("--seed", type=int, default=42,
                       help="Семя для воспроизводимости")
    
    # Параметры сохранения
    parser.add_argument("--save_path", type=str, default="best_deeplob_like.pt",
                       help="Путь для сохранения модели")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Выводить подробную информацию")
    
    return parser.parse_args()


def setup_seed(seed: int):
    """Настройка семени для воспроизводимости"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Основная функция"""
    args = parse_args()
    
    # Настройка семени
    setup_seed(args.seed)
    
    # Создание конфигурации
    config = Config.from_args(args)
    
    print(f"Устройство: {config.training.device}")
    print(f"Семя: {args.seed}")
    print(f"Дни обучения: {config.data.train_dates}")
    print(f"День валидации: {config.data.val_date}")
    print(f"День тестирования: {config.data.test_date}")
    print(f"Эпохи на день: {config.training.epochs}")
    print(f"Размер батча: {config.training.batch_size}")
    print(f"Функция потерь: {'Cost-Sensitive Loss' if config.model.use_cost_sensitive else 'Focal Loss'}")
    print()
    
    # Загрузка и подготовка данных для обучения
    print("Загрузка данных для обучения...")
    train_data_loaders = []
    
    for i, train_date in enumerate(config.data.train_dates):
        print(f"\nОбработка дня обучения {i+1}/{len(config.data.train_dates)}: {train_date}")
        
        # Загрузка данных
        features_file = config.data.get_features_file(train_date)
        prices_file = config.data.get_prices_file(train_date)
        X, ms, mid = load_data(features_file, prices_file)
        
        # Построение меток и окон
        y_all = build_barrier_labels(
            ms, mid, 
            tick_size=config.model.tick_size,
            theta_ticks=config.model.theta_ticks, 
            horizon_sec=config.model.horizon_sec
        )
        
        # Фильтрация валидных меток
        valid = (y_all != -1)
        Xv, yv, msv, midv = X[valid], y_all[valid], ms[valid], mid[valid]
        
        # Создание окон
        Xwin, end_idx = make_windows(Xv, config.model.window_length)
        y_win = yv[end_idx]
        
        print(f"  Классы и счётчики: {np.unique(y_win, return_counts=True)}")
        print(f"  Окна: {Xwin.shape} | метки: {y_win.shape}")
        
        # Создание DataLoader для этого дня
        train_ds = NpWindowDataset(Xwin, y_win)
        base_sampler = BalancedBatchSampler(
            y_win, 
            batch_size=config.training.batch_size, 
            num_classes=config.model.num_classes, 
            seed=config.training.seed
        )
        batch_sampler = BalancedBatchBatchSampler(base_sampler, config.training.batch_size)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler)
        train_data_loaders.append(train_dl)
    
    # Загрузка данных для валидации
    print(f"\nЗагрузка данных для валидации: {config.data.val_date}")
    val_features_file = config.data.get_features_file(config.data.val_date)
    val_prices_file = config.data.get_prices_file(config.data.val_date)
    X_val, ms_val, mid_val = load_data(val_features_file, val_prices_file)
    
    y_val_all = build_barrier_labels(
        ms_val, mid_val, 
        tick_size=config.model.tick_size,
        theta_ticks=config.model.theta_ticks, 
        horizon_sec=config.model.horizon_sec
    )
    
    valid_val = (y_val_all != -1)
    Xv_val, yv_val = X_val[valid_val], y_val_all[valid_val]
    Xwin_val, end_idx_val = make_windows(Xv_val, config.model.window_length)
    y_win_val = yv_val[end_idx_val]
    
    val_ds = NpWindowDataset(Xwin_val, y_win_val)
    val_dl = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)
    
    print(f"Валидация: {Xwin_val.shape} | метки: {y_win_val.shape}")
    
    # Загрузка данных для тестирования
    print(f"\nЗагрузка данных для тестирования: {config.data.test_date}")
    test_features_file = config.data.get_features_file(config.data.test_date)
    test_prices_file = config.data.get_prices_file(config.data.test_date)
    X_test, ms_test, mid_test = load_data(test_features_file, test_prices_file)
    
    y_test_all = build_barrier_labels(
        ms_test, mid_test, 
        tick_size=config.model.tick_size,
        theta_ticks=config.model.theta_ticks, 
        horizon_sec=config.model.horizon_sec
    )
    
    valid_test = (y_test_all != -1)
    Xv_test, yv_test = X_test[valid_test], y_test_all[valid_test]
    Xwin_test, end_idx_test = make_windows(Xv_test, config.model.window_length)
    y_win_test = yv_test[end_idx_test]
    
    test_ds = NpWindowDataset(Xwin_test, y_win_test)
    test_dl = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    
    print(f"Тестирование: {Xwin_test.shape} | метки: {y_win_test.shape}")
    print()
    
    # Создание модели
    print("Создание модели...")
    model, criterion = create_model(config.model)
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, config.training)
    
    model = model.to(config.training.device)
    print(f"Модель создана: {sum(p.numel() for p in model.parameters())} параметров")
    print()
    
    # Создание тренера
    trainer = create_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=config.training.device,
        grad_clip_norm=config.training.grad_clip_norm
    )
    
    # Последовательное обучение на нескольких днях
    print("Начало последовательного обучения...")
    training_results = trainer.train_sequential_days(
        train_data_loaders=train_data_loaders,
        val_loader=val_dl,
        epochs_per_day=config.training.epochs,
        save_path_template="model_day_{}.pt",
        verbose=args.verbose
    )
    
    print(f"\nЛучший macro-F1 на валидации: {training_results['best_score']:.4f}")
    
    # Тестирование
    print("\nТестирование...")
    test_results = trainer.test(test_dl)
    
    # Сохранение финальной модели
    trainer.save_model(args.save_path)
    
    print(f"\nОбучение завершено!")
    print(f"Финальная модель сохранена: {os.path.abspath(args.save_path)}")
    print(f"Промежуточные модели сохранены: model_day_1.pt, model_day_2.pt, ...")


if __name__ == "__main__":
    main()
