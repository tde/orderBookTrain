# Быстрое решение ошибки MemoryError

## Проблема
```
MemoryError: Unable to allocate 28.6 GiB for an array with shape (151544, 240, 211)
```

## ⚡ БЫСТРОЕ РЕШЕНИЕ

Замените в вашем коде:

```python
# СТАРЫЙ КОД (требует 28.6 ГБ):
from tools import make_windows
from dataset import NpWindowDataset

Xwin, end_idx = make_windows(Xv, config.model.window_length)
y_win = yv[end_idx]
train_ds = NpWindowDataset(Xwin, y_win)
```

НА:

```python
# НОВЫЙ КОД (требует ~0.2 ГБ):
from dataset import LazyWindowDataset

train_ds = LazyWindowDataset(Xv, yv, config.model.window_length)
y_win = yv[config.model.window_length-1:]
```

## ✅ Готово!

Ноутбук `notebooks/pipeline.ipynb` уже обновлен и готов к запуску.

## 📚 Дополнительная информация

- Подробное объяснение: `РЕШЕНИЕ_MEMORYERROR.md`
- Все варианты решений: `MEMORY_OPTIMIZATION_GUIDE.md`
- Обновленный код: `src/dataset.py` (новый класс `LazyWindowDataset`)

