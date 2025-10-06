# Исправление ошибки RuntimeError на Windows

## Проблема
```
RuntimeError: Given groups=1, weight of size [128, 138, 1], 
expected input[512, 211, 240] to have 138 channels, but got 211 channels instead
```

## Решение

### Файл 1: src/config.py

Найдите класс `ModelConfig` (строка ~44) и добавьте параметр `input_features`:

```python
@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Параметры задачи
    tick_size: float = 1.0
    theta_ticks: int = 5
    horizon_sec: float = 2.0
    window_length: int = 240
    
    # Архитектура модели
    input_features: int = 211        # ← ДОБАВЬТЕ ЭТУ СТРОКУ
    hidden_size: int = 128
    num_classes: int = 3
    groups: int = 8
    dropout: float = 0.3
    
    # Loss function
    use_cost_sensitive_focal: bool = True
    use_cost_sensitive: bool = False
    focal_alpha: list = None
    focal_gamma: float = 4.0
    focal_alpha_weight: float = 0.5
    cost_weight: float = 2.0
    
    def __post_init__(self):
        if self.focal_alpha is None:
            self.focal_alpha = [22.0, 1.0, 22.0]
```

### Файл 2: src/model.py

Найдите функцию `create_model` (строка ~74) и измените параметр `input_features`:

```python
def create_model(config) -> tuple[nn.Module, nn.Module]:
    """
    Создать модель и функцию потерь
    
    Args:
        config: Конфигурация модели
        
    Returns:
        Кортеж (модель, функция потерь)
    """
    model = DeepLOBLike(
        input_features=getattr(config, 'input_features', 211),  # ← ИЗМЕНИТЕ ЭТУ СТРОКУ (было 138)
        num_classes=config.num_classes,
        hidden_size=config.hidden_size,
        groups=config.groups,
        dropout=config.dropout
    )
    
    # Создание функции потерь
    # ... остальной код без изменений
```

## После исправления

1. Сохраните оба файла
2. Перезапустите Jupyter Notebook kernel
3. Запустите notebook заново

Ошибка должна исчезнуть!

## Дополнительно: Если у вас другое количество признаков

Если ваши данные имеют не 211 признаков, а другое количество, измените значение:

```python
input_features: int = <ваше_количество_признаков>
```

Узнать количество признаков можно так:

```python
import numpy as np
X = np.load('путь_к_features_файлу.npy')
print(f"Количество признаков: {X.shape[1]}")
```

