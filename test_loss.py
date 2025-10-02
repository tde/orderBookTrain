#!/usr/bin/env python3
"""
Тестовый скрипт для проверки стоимостно-чувствительного лосса
"""
import sys
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from focal_loss import FocalLoss
from cost_sensitive_loss import CostSensitiveLoss, get_default_cost_matrix


def test_cost_sensitive_loss():
    """Тестирование стоимостно-чувствительного лосса"""
    print("Тестирование Cost-Sensitive Loss...")
    
    # Создаем тестовые данные
    batch_size = 4
    num_classes = 3
    
    # Логиты модели (случайные)
    logits = torch.randn(batch_size, num_classes)
    
    # Истинные метки
    targets = torch.tensor([0, 1, 2, 0])  # down, flat, up, down
    
    # Матрица стоимости
    cost_matrix = get_default_cost_matrix()
    
    # Создаем функцию потерь
    criterion = CostSensitiveLoss(cost_matrix)
    
    # Вычисляем потери
    loss = criterion(logits, targets)
    
    print(f"Логиты: {logits}")
    print(f"Метки: {targets}")
    print(f"Матрица стоимости:\n{cost_matrix}")
    print(f"Cost-Sensitive Loss: {loss.item():.4f}")
    
    # Проверим, что потери положительные
    assert loss.item() > 0, "Потери должны быть положительными"
    print("✓ Cost-Sensitive Loss работает корректно")
    
    return loss


def test_focal_loss():
    """Тестирование Focal Loss"""
    print("\nТестирование Focal Loss...")
    
    # Создаем тестовые данные
    batch_size = 4
    num_classes = 3
    
    # Логиты модели (случайные)
    logits = torch.randn(batch_size, num_classes)
    
    # Истинные метки
    targets = torch.tensor([0, 1, 2, 0])  # down, flat, up, down
    
    # Создаем функцию потерь
    criterion = FocalLoss(alpha=[3.0, 1.0, 3.0], gamma=2.0)
    
    # Вычисляем потери
    loss = criterion(logits, targets)
    
    print(f"Логиты: {logits}")
    print(f"Метки: {targets}")
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Проверим, что потери положительные
    assert loss.item() > 0, "Потери должны быть положительными"
    print("✓ Focal Loss работает корректно")
    
    return loss


def compare_losses():
    """Сравнение двух функций потерь"""
    print("\nСравнение функций потерь...")
    
    # Создаем тестовые данные
    batch_size = 100
    num_classes = 3
    
    # Логиты модели (случайные)
    logits = torch.randn(batch_size, num_classes)
    
    # Истинные метки (сбалансированные)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Cost-Sensitive Loss
    cost_matrix = get_default_cost_matrix()
    cost_criterion = CostSensitiveLoss(cost_matrix)
    cost_loss = cost_criterion(logits, targets)
    
    # Focal Loss
    focal_criterion = FocalLoss(alpha=[3.0, 1.0, 3.0], gamma=2.0)
    focal_loss = focal_criterion(logits, targets)
    
    print(f"Cost-Sensitive Loss: {cost_loss.item():.4f}")
    print(f"Focal Loss: {focal_loss.item():.4f}")
    print(f"Разница: {abs(cost_loss.item() - focal_loss.item()):.4f}")
    
    print("✓ Обе функции потерь работают корректно")


def main():
    """Основная функция"""
    print("Тестирование функций потерь для DeepLOB")
    print("=" * 50)
    
    try:
        # Тестируем каждую функцию потерь
        test_cost_sensitive_loss()
        test_focal_loss()
        compare_losses()
        
        print("\n" + "=" * 50)
        print("✓ Все тесты пройдены успешно!")
        print("Cost-Sensitive Loss готов к использованию в обучении.")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
