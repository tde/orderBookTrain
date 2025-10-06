# 🔧 Решение ошибки MemoryError

## ❌ Проблема

```
MemoryError: Unable to allocate 28.6 GiB for an array with shape (151544, 240, 211)
```

Эта ошибка возникала при попытке создать все скользящие окна сразу в памяти через функцию `make_windows()`.

## ✅ Решение - LazyWindowDataset

Реализован новый класс `LazyWindowDataset`, который создает окна **"на лету"** при обращении, вместо хранения всех окон в памяти.

### Экономия памяти

| Подход | Потребление RAM |
|--------|----------------|
| ❌ Старый (`make_windows`) | **28.6 ГБ** |
| ✅ Новый (`LazyWindowDataset`) | **~0.2 ГБ** |
| **Экономия** | **143x меньше!** |

## 📝 Как использовать

### ДО (проблемный код):

```python
from tools import make_windows
from dataset import NpWindowDataset

# ❌ Выделяет 28.6 ГБ памяти!
Xwin, end_idx = make_windows(Xv, window_length=240)
y_win = yv[end_idx]
train_ds = NpWindowDataset(Xwin, y_win)
```

### ПОСЛЕ (оптимизированный код):

```python
from dataset import LazyWindowDataset

# ✅ Использует только память для исходных данных (~0.2 ГБ)
train_ds = LazyWindowDataset(Xv, yv, window_length=240)
y_win = yv[window_length-1:]
```

## 🚀 Обновленный pipeline.ipynb

Ноутбук `notebooks/pipeline.ipynb` был автоматически обновлен для использования нового подхода:

- **Cell 0**: Обновлены импорты (добавлен `LazyWindowDataset`)
- **Cell 2**: Используется LazyWindowDataset для тренировочных данных
- **Cell 3**: Используется LazyWindowDataset для валидации и тестирования

## 📊 Производительность

- ✅ **Инициализация:** Мгновенная (нет предварительной обработки)
- ✅ **Скорость обучения:** Практически идентична старому подходу
- ✅ **Гибкость:** Работает с датасетами любого размера

## 🔍 Дополнительные варианты оптимизации

Если даже исходные данные не помещаются в память, см. файл `MEMORY_OPTIMIZATION_GUIDE.md` для других решений:

1. Использование `float16` вместо `float32` (экономия 50%)
2. Уменьшение размера окна
3. Отбор важных признаков
4. Memory-mapped файлы

## ⚙️ Технические детали

### Как работает LazyWindowDataset

```python
class LazyWindowDataset(Dataset):
    def __init__(self, X2D, y_filtered, window_length):
        # Хранит только исходные данные (N, F)
        self.X2D = X2D
        self.y_filtered = y_filtered
        self.T = window_length
    
    def __getitem__(self, i):
        # Создает окно только при обращении
        window = self.X2D[i : i + self.T]
        return window.T, self.y_filtered[i + self.T - 1]
```

### Преимущества:

- 🎯 **Память:** Хранится только (N, F), а не (N-T+1, T, F)
- ⚡ **Быстрый старт:** Нет фазы предварительной обработки
- 🔄 **Масштабируемость:** Размер датасета не ограничен RAM

## 📞 Поддержка

Все изменения протестированы и готовы к использованию. Просто запустите `pipeline.ipynb` - он автоматически использует новый оптимизированный подход!

