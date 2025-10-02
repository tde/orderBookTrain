#!/usr/bin/env python3
"""
Скрипт для быстрого запуска обучения с разными конфигурациями
"""
import subprocess
import sys
from pathlib import Path


def run_training(config_name: str, **kwargs):
    """Запуск обучения с заданной конфигурацией"""
    cmd = [sys.executable, "train.py"]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Запуск конфигурации '{config_name}'...")
    print(f"Команда: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print(f"Конфигурация '{config_name}' выполнена успешно!")
    else:
        print(f"Ошибка в конфигурации '{config_name}'!")
    
    print("=" * 50)
    return result.returncode == 0


def main():
    """Основная функция с различными конфигурациями"""
    
    # Базовая конфигурация
    base_config = {
        "train_date": "2025-09-22",
        "epochs": 15,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "seed": 42
    }
    
    # Конфигурации для экспериментов
    configs = {
        "baseline": base_config,
        
        "smaller_model": {
            **base_config,
            "hidden_size": 64,
            "dropout": 0.2
        },
        
        "larger_model": {
            **base_config,
            "hidden_size": 256,
            "dropout": 0.4
        },
        
        "faster_training": {
            **base_config,
            "epochs": 10,
            "batch_size": 1024,
            "learning_rate": 2e-3
        },
        
        "slower_training": {
            **base_config,
            "epochs": 25,
            "batch_size": 256,
            "learning_rate": 5e-4
        },
        
        "different_horizon": {
            **base_config,
            "horizon_sec": 1.5,
            "theta_ticks": 3
        },
        
        "focal_loss": {
            **base_config,
            "use_focal_loss": True
        },
        
        "cost_sensitive": {
            **base_config,
            "use_cost_sensitive": True
        }
    }
    
    print("Доступные конфигурации:")
    for i, name in enumerate(configs.keys(), 1):
        print(f"{i}. {name}")
    
    print("\nВыберите конфигурацию (номер или 'all' для всех):")
    choice = input().strip()
    
    if choice.lower() == "all":
        # Запуск всех конфигураций
        for name, config in configs.items():
            success = run_training(name, **config)
            if not success:
                print(f"Остановка из-за ошибки в конфигурации '{name}'")
                break
    else:
        try:
            config_num = int(choice) - 1
            config_names = list(configs.keys())
            if 0 <= config_num < len(config_names):
                config_name = config_names[config_num]
                config = configs[config_name]
                run_training(config_name, **config)
            else:
                print("Неверный номер конфигурации!")
        except ValueError:
            print("Введите номер конфигурации или 'all'!")


if __name__ == "__main__":
    main()
