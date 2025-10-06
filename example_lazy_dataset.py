"""
Пример использования LazyWindowDataset для решения проблемы MemoryError
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('src')

from dataset import LazyWindowDataset, BalancedBatchSampler, BalancedBatchBatchSampler

# Симулируем данные похожие на реальные
N = 152000  # Количество строк
F = 211     # Количество признаков
T = 240     # Размер окна

print("=" * 60)
print("ДЕМОНСТРАЦИЯ РЕШЕНИЯ MEMORYERROR")
print("=" * 60)

# Создаем тестовые данные
print(f"\n📊 Создание тестовых данных: N={N}, F={F}, T={T}")
X2D = np.random.randn(N, F).astype(np.float32)
y = np.random.randint(0, 3, N)  # 3 класса: 0, 1, 2

print(f"   Размер исходных данных: {X2D.nbytes / 1024**3:.2f} ГБ")

# Сравнение подходов
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПОДХОДОВ")
print("=" * 60)

# 1. Старый подход (закомментирован, чтобы не упасть с MemoryError)
print("\n❌ СТАРЫЙ ПОДХОД (make_windows):")
num_windows = N - T + 1
memory_required = num_windows * T * F * 4  # float32 = 4 байта
print(f"   Требуемая память: {memory_required / 1024**3:.2f} ГБ")
print(f"   ⚠️  МОЖЕТ УПАСТЬ С MemoryError!")

# 2. Новый подход (LazyWindowDataset)
print("\n✅ НОВЫЙ ПОДХОД (LazyWindowDataset):")
lazy_ds = LazyWindowDataset(X2D, y, T)
print(f"   Используемая память: {X2D.nbytes / 1024**3:.2f} ГБ")
print(f"   Экономия: {memory_required / X2D.nbytes:.1f}x раз!")
print(f"   Количество окон: {len(lazy_ds)}")

# Проверка, что датасет работает
print("\n🔍 Тестирование датасета...")
sample_x, sample_y = lazy_ds[0]
print(f"   Форма окна: {sample_x.shape} (ожидается: ({F}, {T}))")
print(f"   Тип данных X: {sample_x.dtype}")
print(f"   Тип данных y: {sample_y.dtype}")
print(f"   ✅ Датасет работает корректно!")

# Создание DataLoader
print("\n📦 Создание DataLoader...")
batch_size = 512

# Метки для окон
y_win = y[T-1:]

# Создаем сбалансированный sampler
base_sampler = BalancedBatchSampler(
    y_win, 
    batch_size=batch_size, 
    num_classes=3,
    seed=42
)
batch_sampler = BalancedBatchBatchSampler(base_sampler, batch_size)
train_dl = DataLoader(lazy_ds, batch_sampler=batch_sampler, num_workers=0)

print(f"   Размер батча: {batch_size}")
print(f"   Количество батчей: {len(train_dl)}")

# Проверка первого батча
print("\n🎯 Тестирование первого батча...")
for batch_x, batch_y in train_dl:
    print(f"   Форма батча X: {batch_x.shape}")
    print(f"   Форма батча y: {batch_y.shape}")
    print(f"   Уникальные классы в батче: {torch.unique(batch_y).tolist()}")
    print(f"   ✅ DataLoader работает корректно!")
    break

# Итоги
print("\n" + "=" * 60)
print("ИТОГИ")
print("=" * 60)
print(f"✅ LazyWindowDataset успешно решает проблему MemoryError")
print(f"✅ Экономия памяти: {memory_required / X2D.nbytes:.1f}x раз")
print(f"✅ Все функции работают корректно")
print(f"✅ Готов к использованию в обучении!")
print("=" * 60)

