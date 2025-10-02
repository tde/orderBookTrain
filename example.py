#!/usr/bin/env python3
"""
Пример скрипта для демонстрации последовательного обучения
"""
import subprocess
import sys
from pathlib import Path


def run_example():
    """Запуск примера последовательного обучения"""
    
    # Пример с двумя днями обучения
    cmd = [
        sys.executable, "train.py",
        "--train_dates", "2025-09-22,2025-09-23",
        "--val_date", "2025-09-24", 
        "--test_date", "2025-09-25",
        "--epochs", "5",  # Меньше эпох для быстрого тестирования
        "--batch_size", "256",
        "--save_path", "example_model.pt"
    ]
    
    print("Запуск примера последовательного обучения...")
    print(f"Команда: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print("\nПример выполнен успешно!")
        print("Созданные файлы:")
        print("- model_day_1.pt (после первого дня обучения)")
        print("- model_day_2.pt (после второго дня обучения)")
        print("- example_model.pt (финальная модель)")
        
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении: {e}")
        return False
    except FileNotFoundError:
        print("Файл train.py не найден. Убедитесь, что вы находитесь в правильной директории.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_example()
    sys.exit(0 if success else 1)
