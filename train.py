#!/usr/bin/env python3
"""
Основной скрипт для обучения модели DeepLOB
Все параметры настраиваются в src/config.py
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from load_data import load_data
from tools import build_barrier_labels
from model import create_model, create_optimizer_and_scheduler
from dataset import LazyWindowDataset, BalancedBatchSampler, BalancedBatchBatchSampler
from trainer import create_trainer
from torch.utils.data import DataLoader


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
    # Создание конфигурации из config.py
    config = Config.default()
    
    # Настройка параметров (такие же как в pipeline.ipynb)
    config.data.train_dates = ["2025-09-22", "2025-09-23", "2025-09-25", "2025-09-26"]
    config.data.val_date = "2025-09-29"
    config.data.test_date = "2025-09-30"
    config.data.data_folder = "./data/npy"
    
    config.model.tick_size = 1.0
    config.model.theta_ticks = 5
    config.model.horizon_sec = 2.0
    config.model.window_length = 240
    config.model.use_cost_sensitive_focal = True
    
    config.training.epochs = 15
    
    # Настройка семени
    setup_seed(config.training.seed)
    
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
    print(f"Устройство: {config.training.device}")
    print()
    
    # Вывод конфигурации
    config.print_config()
    
    # Загрузка и подготовка данных для обучения (MEMORY-EFFICIENT VERSION)
    print("Загрузка данных для обучения...")
    print("⚡ Используется LazyWindowDataset - экономия памяти!")
    train_data_loaders = []
    
    for i, train_date in enumerate(config.data.train_dates):
        print(f"\nОбработка дня обучения {i+1}/{len(config.data.train_dates)}: {train_date}")
        
        # Загрузка данных
        features_file = config.data.get_features_file(train_date)
        print(f"Файл признаков: {features_file}")
        prices_file = config.data.get_prices_file(train_date)
        X, ms, mid = load_data(features_file, prices_file)
        
        # Построение меток
        y_all = build_barrier_labels(
            ms, mid, 
            tick_size=config.model.tick_size,
            theta_ticks=config.model.theta_ticks, 
            horizon_sec=config.model.horizon_sec
        )
        
        # Фильтрация валидных меток
        valid = (y_all != -1)
        Xv, yv = X[valid], y_all[valid]
        
        # LAZY подход - окна создаются на лету, не храним все в памяти
        train_ds = LazyWindowDataset(Xv, yv, config.model.window_length)
        
        # Метки для окон (метка окна = метка последнего элемента)
        y_win = yv[config.model.window_length-1:]
        
        print(f"  Классы и счётчики: {np.unique(y_win, return_counts=True)}")
        print(f"  Датасет: {len(train_ds)} окон | метки: {y_win.shape}")
        print(f"  💾 Память: ~{Xv.nbytes / 1024**3:.2f} ГБ (вместо ~{len(train_ds) * config.model.window_length * Xv.shape[1] * 4 / 1024**3:.2f} ГБ)")
        
        # Создание DataLoader для этого дня
        base_sampler = BalancedBatchSampler(
            y_win, 
            batch_size=config.training.batch_size, 
            num_classes=config.model.num_classes, 
            seed=config.training.seed
        )
        batch_sampler = BalancedBatchBatchSampler(base_sampler, config.training.batch_size)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
        train_data_loaders.append(train_dl)
    
    # Загрузка данных для валидации (MEMORY-EFFICIENT VERSION)
    print(f"\n⚡ Загрузка данных для валидации: {config.data.val_date}")
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
    
    # LAZY подход
    val_ds = LazyWindowDataset(Xv_val, yv_val, config.model.window_length)
    y_win_val = yv_val[config.model.window_length-1:]
    val_dl = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    
    print(f"Валидация: {len(val_ds)} окон | метки: {y_win_val.shape}")
    print(f"  💾 Память: ~{Xv_val.nbytes / 1024**3:.2f} ГБ")
    
    # Загрузка данных для тестирования (MEMORY-EFFICIENT VERSION)
    print(f"\n⚡ Загрузка данных для тестирования: {config.data.test_date}")
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
    
    # LAZY подход
    test_ds = LazyWindowDataset(Xv_test, yv_test, config.model.window_length)
    y_win_test = yv_test[config.model.window_length-1:]
    test_dl = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    
    print(f"Тестирование: {len(test_ds)} окон | метки: {y_win_test.shape}")
    print(f"  💾 Память: ~{Xv_test.nbytes / 1024**3:.2f} ГБ")
    print()
    
    print("Все DataLoader'ы созданы успешно!")
    print(f"Количество дней обучения: {len(train_data_loaders)}")
    print(f"Валидационный DataLoader: {len(val_dl)} батчей")
    print(f"Тестовый DataLoader: {len(test_dl)} батчей")
    print()
    
    # Создание модели
    model, criterion = create_model(config.model)
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, config.training)
    
    model = model.to(config.training.device)
    print(f"Модель создана: {sum(p.numel() for p in model.parameters())} параметров")
    print(f"Устройство: {config.training.device}")
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
        verbose=True
    )
    
    print(f"\nЛучший macro-F1 на валидации: {training_results['best_score']:.4f}")
    
    # Тестирование
    test_results = trainer.test(test_dl)
    
    # Сохранение финальной модели
    model_path = "best_deeplob_like.pt"
    trainer.save_model(model_path)
    
    print(f"\nОбучение завершено!")
    print(f"Финальная модель сохранена: {os.path.abspath(model_path)}")
    print(f"Промежуточные модели сохранены: model_day_1.pt, model_day_2.pt, ...")


if __name__ == "__main__":
    main()
