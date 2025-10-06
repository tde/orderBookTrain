"""
Диагностический скрипт для проверки модели и данных
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from config import Config
from model import create_model
from dataset import LazyWindowDataset
from torch.utils.data import DataLoader

print("=" * 60)
print("ДИАГНОСТИКА МОДЕЛИ И ДАННЫХ")
print("=" * 60)

# 1. Проверка конфигурации
print("\n1. Проверка конфигурации...")
config = Config.default()
print(f"   ✓ input_features: {config.model.input_features}")
print(f"   ✓ hidden_size: {config.model.hidden_size}")
print(f"   ✓ window_length: {config.model.window_length}")

# 2. Создание модели
print("\n2. Создание модели...")
model, criterion = create_model(config.model)
print(f"   ✓ Модель создана")
print(f"   ✓ Параметров: {sum(p.numel() for p in model.parameters()):,}")

# 3. Создание тестовых данных
print("\n3. Создание тестовых данных...")
N = 1000  # Количество сэмплов
F = config.model.input_features  # Признаки
T = config.model.window_length  # Длина окна

X_test = np.random.randn(N, F).astype(np.float32)
y_test = np.random.randint(0, 3, N).astype(np.int64)

print(f"   ✓ X_test shape: {X_test.shape}")
print(f"   ✓ y_test shape: {y_test.shape}")
print(f"   ✓ X_test dtype: {X_test.dtype}")
print(f"   ✓ y_test dtype: {y_test.dtype}")

# 4. Создание LazyWindowDataset
print("\n4. Создание LazyWindowDataset...")
dataset = LazyWindowDataset(X_test, y_test, T)
print(f"   ✓ Dataset size: {len(dataset)}")

# 5. Проверка одного сэмпла
print("\n5. Проверка одного сэмпла...")
x_sample, y_sample = dataset[0]
print(f"   ✓ x_sample shape: {x_sample.shape} (ожидается: ({F}, {T}))")
print(f"   ✓ y_sample shape: {y_sample.shape} (ожидается: ())")
print(f"   ✓ x_sample dtype: {x_sample.dtype}")
print(f"   ✓ y_sample dtype: {y_sample.dtype}")
print(f"   ✓ x_sample range: [{x_sample.min():.2f}, {x_sample.max():.2f}]")

# 6. Создание DataLoader
print("\n6. Создание DataLoader...")
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"   ✓ DataLoader создан")
print(f"   ✓ Количество батчей: {len(loader)}")

# 7. Проверка одного батча
print("\n7. Проверка одного батча...")
for xb, yb in loader:
    print(f"   ✓ Batch x shape: {xb.shape} (ожидается: (32, {F}, {T}))")
    print(f"   ✓ Batch y shape: {yb.shape} (ожидается: (32,))")
    print(f"   ✓ Batch x dtype: {xb.dtype}")
    print(f"   ✓ Batch y dtype: {yb.dtype}")
    print(f"   ✓ Batch x range: [{xb.min():.2f}, {xb.max():.2f}]")
    break

# 8. Forward pass через модель
print("\n8. Forward pass через модель...")
model.eval()
with torch.no_grad():
    try:
        logits = model(xb)
        print(f"   ✓ Forward pass успешен!")
        print(f"   ✓ Output shape: {logits.shape} (ожидается: (32, 3))")
        print(f"   ✓ Output range: [{logits.min():.2f}, {logits.max():.2f}]")
        
        # Проверка softmax
        probs = torch.softmax(logits, dim=1)
        print(f"   ✓ Probabilities sum: {probs.sum(dim=1).mean():.4f} (должно быть ~1.0)")
        
    except Exception as e:
        print(f"   ✗ ОШИБКА в forward pass:")
        print(f"      {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# 9. Проверка loss
print("\n9. Проверка loss...")
try:
    model.train()
    logits = model(xb)
    loss = criterion(logits, yb)
    print(f"   ✓ Loss вычислен: {loss.item():.4f}")
    
    # Проверка backward
    loss.backward()
    print(f"   ✓ Backward pass успешен!")
    
    # Проверка градиентов
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"   ✗ NaN gradient в {name}")
                has_nan_grad = True
    
    if not has_nan_grad:
        print(f"   ✓ Градиенты в порядке (нет NaN)")
        
except Exception as e:
    print(f"   ✗ ОШИБКА в loss/backward:")
    print(f"      {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 10. Итоги
print("\n" + "=" * 60)
print("ИТОГИ ДИАГНОСТИКИ")
print("=" * 60)
print("Если все пункты прошли успешно (✓), модель готова к обучению.")
print("Если есть ошибки (✗), проверьте:")
print("  1. Размерности данных")
print("  2. Типы данных (float32, int64)")
print("  3. Наличие NaN в данных")
print("  4. Параметры модели в config.py")
print("=" * 60)

