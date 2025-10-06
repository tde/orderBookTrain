# Решения проблемы MemoryError при создании окон

## Проблема
При вызове `make_windows()` для создания скользящих окон возникает ошибка:
```
MemoryError: Unable to allocate 28.6 GiB for an array with shape (151544, 240, 211)
```

## Решения (от лучшего к простейшему)

### 1. ✅ LazyWindowDataset (РЕКОМЕНДУЕТСЯ)

**Самое эффективное решение** - создание окон "на лету" без хранения всех в памяти.

**Пример использования в notebook:**

```python
from dataset import LazyWindowDataset
from torch.utils.data import DataLoader

# После фильтрации валидных меток
valid = (y_all != -1)
Xv, yv = X[valid], y_all[valid]

# ВМЕСТО make_windows + NpWindowDataset используем LazyWindowDataset
train_ds = LazyWindowDataset(Xv, yv, config.model.window_length)

# Создаем метки для окон (метка = последний элемент окна)
y_win = yv[config.model.window_length-1:]

# Создаем DataLoader с сэмплером
from dataset import BalancedBatchSampler, BalancedBatchBatchSampler

base_sampler = BalancedBatchSampler(
    y_win, 
    batch_size=config.training.batch_size,
    num_classes=config.model.num_classes,
    seed=config.training.seed
)
batch_sampler = BalancedBatchBatchSampler(base_sampler, config.training.batch_size)
train_dl = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
```

**Обновленный пример для pipeline.ipynb Cell 2:**

```python
# Загрузка и подготовка данных для обучения
print("Загрузка данных для обучения...")
train_data_loaders = []

for i, train_date in enumerate(config.data.train_dates):
    print(f"\nОбработка дня обучения {i+1}/{len(config.data.train_dates)}: {train_date}")
    
    # Загрузка данных
    features_file = config.data.get_features_file(train_date)
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
    
    # LAZY ПОДХОД - без создания всех окон в памяти
    train_ds = LazyWindowDataset(Xv, yv, config.model.window_length)
    y_win = yv[config.model.window_length-1:]
    
    print(f"  Классы и счётчики: {np.unique(y_win, return_counts=True)}")
    print(f"  Датасет: {len(train_ds)} окон | метки: {y_win.shape}")
    
    # Создание DataLoader
    base_sampler = BalancedBatchSampler(
        y_win, 
        batch_size=config.training.batch_size, 
        num_classes=config.model.num_classes, 
        seed=config.training.seed
    )
    batch_sampler = BalancedBatchBatchSampler(base_sampler, config.training.batch_size)
    train_dl = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
    train_data_loaders.append(train_dl)
```

**Преимущества:**
- ✅ Экономия памяти: ~28.6 ГБ → ~200 МБ (только исходные данные)
- ✅ Работает с любым размером датасета
- ✅ Быстрая инициализация
- ✅ Окна создаются только при обращении

**Недостатки:**
- Немного медленнее при первой эпохе (но незначительно)

---

### 2. ⚙️ Использование float16 вместо float32

Уменьшаем размер данных в 2 раза:

```python
from load_data import load_data

# В load_data.py изменить dtype при загрузке
X = np.load(features_file).astype(np.float16)  # вместо float32
```

**Экономия:** 28.6 ГБ → 14.3 ГБ

**Недостаток:** Может снизить точность модели

---

### 3. 🔧 Уменьшение размера окна

```python
config.model.window_length = 120  # вместо 240
```

**Экономия:** 28.6 ГБ → 14.3 ГБ

**Недостаток:** Модель видит меньше исторических данных

---

### 4. 📊 Отбор важных признаков

Уменьшить количество признаков с 211 до 50-100:

```python
# Анализ важности признаков (отдельная задача)
important_features = [0, 1, 2, ...]  # индексы важных признаков
X_filtered = X[:, important_features]
```

**Экономия:** Пропорционально количеству признаков

---

### 5. 🗂️ Chunked обработка (если нужны все окна)

Если по какой-то причине нужно создать все окна в памяти:

```python
from tools import make_windows_chunked

# Вместо make_windows используем chunked версию
Xwin, end_idx = make_windows_chunked(Xv, config.model.window_length, chunk_size=10000)
```

Эта функция все равно создаст все окна, но обработает их частями, что может помочь на некоторых системах.

---

### 6. 💾 Memory-mapped файлы (для очень больших данных)

Хранить окна на диске:

```python
# Создаем memory-mapped массив
Xwin_mmap = np.memmap('windows_temp.dat', dtype='float32', mode='w+', 
                       shape=(num_windows, T, F))

# Заполняем по частям
for i in range(num_windows):
    Xwin_mmap[i] = X[i:i+T]

# Используем как обычный массив
```

---

## Рекомендации

1. **Для продакшена и больших датасетов:** Используйте `LazyWindowDataset` (вариант 1)
2. **Для экспериментов с небольшими данными:** Оригинальный `make_windows` работает нормально
3. **При нехватке RAM даже для исходных данных:** Комбинируйте float16 + LazyWindowDataset
4. **При нехватке GPU памяти:** Уменьшите `batch_size` в конфигурации

## Примерная экономия памяти

| Подход | RAM для окон | Комментарий |
|--------|--------------|-------------|
| Оригинальный | 28.6 ГБ | Все окна в памяти |
| LazyWindowDataset | ~0.2 ГБ | Только исходные данные |
| float16 + оригинальный | 14.3 ГБ | Половина |
| LazyWindowDataset + float16 | ~0.1 ГБ | Минимум |
| Окно 120 вместо 240 | 14.3 ГБ | Половина |

---

## Дополнительные оптимизации

### Использование готовой функции для создания lazy loaders:

```python
from dataset import create_lazy_data_loaders

# Создать все 3 загрузчика сразу
train_dl, val_dl, test_dl = create_lazy_data_loaders(
    X2D_train=Xv_train,
    y_train=yv_train,
    X2D_val=Xv_val,
    y_val=yv_val,
    X2D_test=Xv_test,
    y_test=yv_test,
    window_length=config.model.window_length,
    batch_size=config.training.batch_size,
    num_classes=config.model.num_classes,
    seed=config.training.seed
)
```


