# Сводка всех исправлений

## 🚨 Критические исправления (устраняют NaN loss)

### 1. ✅ Исправлена проблема с device (CPU vs CUDA)
**Файлы:** `src/cost_sensitive_focal_loss.py`, `src/cost_sensitive_loss.py`

**Проблема:** Матрица стоимости была на CPU, а данные на CUDA

**Решение:** Добавлено `.to(targets.device)` перед использованием матрицы

```python
# БЫЛО (вызывало ошибку):
batch_costs = self.cost_matrix[targets]

# СТАЛО (исправлено):
cost_matrix = self.cost_matrix.to(targets.device)
batch_costs = cost_matrix[targets]
```

---

### 2. ✅ Уменьшены параметры loss (предотвращение NaN)
**Файл:** `src/config.py`

**Проблема:** Слишком агрессивные параметры вызывали взрыв градиентов

**Изменения:**
```python
# БЫЛО (вызывало NaN):
focal_gamma = 4.0
focal_alpha = [22.0, 1.0, 22.0]
focal_alpha_weight = 0.5
cost_weight = 2.0

# СТАЛО (стабильно):
focal_gamma = 2.0
focal_alpha = [5.0, 1.0, 5.0]
focal_alpha_weight = 0.25
cost_weight = 0.5
```

---

### 3. ✅ Уменьшен learning rate
**Файл:** `src/config.py`

**Изменение:**
```python
learning_rate: float = 5e-4  # было 1e-3
```

---

### 4. ✅ Добавлена нормализация данных
**Файл:** `src/load_data.py`

**Что добавлено:**
- Z-score нормализация всех признаков
- Предотвращение деления на 0
- Автоматическое применение (по умолчанию `normalize=True`)

```python
mean = np.mean(X, axis=0, keepdims=True)
std = np.std(X, axis=0, keepdims=True)
std = np.where(std < 1e-8, 1.0, std)
X = (X - mean) / std
```

---

### 5. ✅ Добавлены проверки NaN в trainer
**Файл:** `src/trainer.py`

**Что добавлено:**
- Проверка loss на NaN/Inf перед backward
- Проверка градиентов на NaN/Inf
- Skip проблемных батчей вместо краша

```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"⚠️  WARNING: Loss is {loss.item()}, skipping batch")
    continue
```

---

## 🎯 Улучшения архитектуры модели

### 6. ✅ Добавлены Residual Connections
**Файл:** `src/model.py`

**Что добавлено:**
```python
class TemporalConvBlockWithResidual(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv_layers(x)
        return self.act(out + identity)  # ← Residual connection
```

**Преимущества:**
- Решает проблему затухающих градиентов
- Позволяет обучать более глубокие сети

---

### 7. ✅ Больше дилатаций (1, 2, 4, 8, 16)
**Файл:** `src/model.py`

**Было:** 3 блока с дилатациями `[1, 2, 4]`
**Стало:** 5 блоков с дилатациями `[1, 2, 4, 8, 16]`

**Receptive field:**
- Было: ~40 временных шагов
- Стало: ~160 временных шагов (+4x)

---

### 8. ✅ Attention механизм вместо AdaptiveAvgPool
**Файл:** `src/model.py`

**Что добавлено:**
```python
class TemporalAttention(nn.Module):
    # Multi-head attention (4 головы)
    # Адаптивное взвешивание важности временных шагов
```

**Преимущества:**
- Модель сама учится, какие моменты времени важнее
- AdaptiveAvgPool усредняет всё равномерно
- Attention фокусируется на критических событиях

---

## 📊 Ожидаемые результаты

### До исправлений:
```
[01] train_loss=nan acc=0.334 | val F1↓=0.101 F1○=0.000 F1↑=0.000
Confusion matrix:
[[  7409      0      0]    ← модель предсказывает только класс 0
 [124899      0      0]
 [  7503      0      0]]
```

### После исправлений (ожидается):
```
[01] train_loss=0.85 acc=0.65 | val F1↓=0.55 F1○=0.88 F1↑=0.57
Confusion matrix:
[[  4200   1800   1409]    ← модель различает все классы
 [ 12000 105000  7899]
 [  1300   2100   4103]]
```

---

## 🔧 Что нужно применить на Windows

### Файл 1: `src/config.py`
```python
@dataclass
class ModelConfig:
    # Архитектура
    input_features: int = 211        # ← ДОБАВИТЬ
    hidden_size: int = 128
    num_classes: int = 3
    groups: int = 8
    dropout: float = 0.3
    
    # Loss function (КОНСЕРВАТИВНЫЕ параметры)
    use_cost_sensitive_focal: bool = True
    use_cost_sensitive: bool = False
    focal_alpha: list = None
    focal_gamma: float = 2.0               # ← ИЗМЕНИТЬ с 4.0
    focal_alpha_weight: float = 0.25       # ← ИЗМЕНИТЬ с 0.5
    cost_weight: float = 0.5               # ← ИЗМЕНИТЬ с 2.0
    
    def __post_init__(self):
        if self.focal_alpha is None:
            self.focal_alpha = [5.0, 1.0, 5.0]  # ← ИЗМЕНИТЬ с [22.0, 1.0, 22.0]

@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 5e-4           # ← ИЗМЕНИТЬ с 1e-3
    weight_decay: float = 1e-4
```

### Файл 2: `src/model.py`
**Полностью заменить на новую версию** (с residual + attention)

Ключевые изменения:
- Добавлен класс `TemporalConvBlockWithResidual`
- Добавлен класс `TemporalAttention`
- Класс `DeepLOBLike` полностью переписан
- В `create_model` изменено: `input_features=getattr(config, 'input_features', 211)`

### Файл 3: `src/cost_sensitive_focal_loss.py`
Найти строки ~83-85 и изменить:
```python
# БЫЛО:
batch_costs = self.cost_matrix[targets]

# СТАЛО:
cost_matrix = self.cost_matrix.to(targets.device)
batch_costs = cost_matrix[targets]
```

Найти строки ~148-150 (в классе AdaptiveCostSensitiveFocalLoss):
```python
# БЫЛО:
alpha_t = (self.alpha.unsqueeze(0) * targets_one_hot).sum(dim=1)

# СТАЛО:
alpha = self.alpha.to(logits.device)
alpha_t = (alpha.unsqueeze(0) * targets_one_hot).sum(dim=1)
```

### Файл 4: `src/cost_sensitive_loss.py`
Найти строки ~31-33:
```python
# БЫЛО:
C_y = self.C[targets]

# СТАЛО:
C = self.C.to(targets.device)
C_y = C[targets]
```

### Файл 5: `src/load_data.py`
Добавить параметр `normalize=True` и код нормализации (см. полный файл в изменениях)

### Файл 6: `src/trainer.py`
Добавить проверки NaN в методе `train_epoch` (см. полный файл в изменениях)

---

## 📝 Порядок действий

1. **Скопировать все изменения** из Mac на Windows машину
2. **Перезапустить Jupyter Notebook kernel**
3. **Запустить обучение** с новыми параметрами
4. **Мониторить:**
   - Train loss не должен быть NaN
   - Accuracy должен расти
   - Confusion matrix должна показывать все 3 класса

5. **Постепенно увеличивать сложность loss** (если обучение стабильно):
   - После 5 эпох: увеличить `focal_gamma` до 2.5, `focal_alpha` до `[8, 1, 8]`
   - После 10 эпох: увеличить `focal_gamma` до 3.0, `focal_alpha` до `[12, 1, 12]`

---

## ✅ Чек-лист

- [ ] Скопированы все изменения на Windows
- [ ] Перезапущен Jupyter kernel
- [ ] Проверено: train_loss не NaN
- [ ] Проверено: модель предсказывает все 3 класса
- [ ] Confusion matrix показывает не только нули
- [ ] Macro F1 > 0.5 после нескольких эпох

---

## 🆘 Если проблемы остались

1. **Если всё ещё NaN:**
   - Уменьшите learning_rate до 1e-4
   - Уменьшите focal_gamma до 1.5
   - Проверьте, что нормализация данных применяется

2. **Если модель предсказывает только 1 класс:**
   - Проверьте BalancedBatchSampler работает
   - Увеличьте cost_weight до 0.8-1.0
   - Проверьте focal_alpha

3. **Если out of memory:**
   - Уменьшите batch_size до 256
   - Уменьшите hidden_size до 96
   - Уменьшите window_length до 180

Удачи! 🚀

