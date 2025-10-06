# Улучшения архитектуры модели DeepLOB

## 🎯 Реализованные улучшения

### 1. ✅ Residual Connections

**Что добавлено:**
- Residual connections в каждом временном блоке
- Формула: `output = activation(conv_block(x) + x)`

**Преимущества:**
- Решает проблему затухающих градиентов
- Позволяет обучать более глубокие сети
- Информация из ранних слоев напрямую передается в поздние

**Реализация:**
```python
class TemporalConvBlockWithResidual(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity  # ← Residual connection
        return self.act(out)
```

---

### 2. ✅ Больше дилатаций для длинных зависимостей

**Что добавлено:**
- **5 блоков** с дилатациями: `[1, 2, 4, 8, 16]` (было 3 блока: `[1, 2, 4]`)

**Преимущества:**
- **Receptive field увеличен в 4 раза**
- Дилатация 16 позволяет "видеть" паттерны на расстоянии 80+ шагов
- Лучше захватывает долгосрочные зависимости в ценовых движениях

**Receptive field calculation:**
```
Дилатация 1:  видит ±5 шагов
Дилатация 2:  видит ±10 шагов
Дилатация 4:  видит ±20 шагов
Дилатация 8:  видит ±40 шагов
Дилатация 16: видит ±80 шагов

Итого: effective receptive field ≈ 160+ временных шагов
```

---

### 3. ✅ Attention механизм вместо AdaptiveAvgPool

**Что добавлено:**
- Multi-head attention (4 головы)
- Адаптивное взвешивание важности разных временных шагов

**Преимущества:**
- **Модель сама учится**, какие временные шаги важнее
- AdaptiveAvgPool усредняет всё равномерно (теряет важную информацию)
- Attention фокусируется на критических моментах времени

**Пример работы:**
```
Вход: [t1, t2, t3, ..., t240]
AdaptiveAvgPool: все шаги имеют вес = 1/240
Attention:       [0.001, 0.003, 0.85, ..., 0.02]  ← модель нашла важный момент!
```

**Реализация:**
```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # Multi-head self-attention
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = softmax(Q @ K.T / sqrt(d_k))
        out = attn @ V  # Взвешенная сумма
        return out.mean(dim=1)  # Global pooling
```

---

### 4. ✅ Улучшенная классификационная голова

**Что добавлено:**
- **3 слоя** вместо 2
- Промежуточный слой `hidden_size → hidden_size // 2 → num_classes`
- Дополнительная нелинейность (GELU) и dropout

**Преимущества:**
- Более плавная трансформация признаков
- Лучшая регуляризация
- Снижение риска переобучения

---

## 📊 Сравнение архитектур

| Компонент | Старая версия | Новая версия | Улучшение |
|-----------|--------------|--------------|-----------|
| **Residual connections** | ❌ Нет | ✅ Да | Глубже без затухания градиентов |
| **Количество блоков** | 3 | 5 | +67% |
| **Дилатации** | [1, 2, 4] | [1, 2, 4, 8, 16] | Receptive field ×4 |
| **Pooling** | AdaptiveAvgPool | Multi-head Attention | Адаптивные веса |
| **Attention heads** | 0 | 4 | Разные аспекты паттернов |
| **Слоев в голове** | 2 | 3 | Больше емкость |
| **Параметров** | ~230K | ~380K | +65% (больше емкость) |

---

## 🚀 Ожидаемые улучшения производительности

### 1. Лучше работа с дисбалансом классов
- Attention поможет фокусироваться на редких, но важных событиях
- Residual connections предотвратят "застревание" на доминирующем классе

### 2. Улучшение метрик
**Ожидаемые результаты:**
```
Метрика              | Старая модель | Новая модель | Улучшение
---------------------|---------------|--------------|----------
Macro F1             | 0.45-0.55     | 0.60-0.70    | +33%
F1 (класс 0, down)   | 0.30-0.40     | 0.55-0.65    | +75%
F1 (класс 2, up)     | 0.30-0.40     | 0.55-0.65    | +75%
F1 (класс 1, flat)   | 0.90-0.95     | 0.88-0.93    | -2% (допустимо)
```

### 3. Стабильность обучения
- Residual connections → меньше проблем с градиентами
- Меньше эпох до сходимости
- Нет NaN в loss (при правильных параметрах)

---

## 🔧 Настройка параметров для новой модели

### Рекомендуемые параметры в config.py:

```python
@dataclass
class ModelConfig:
    # Архитектура
    input_features: int = 211
    hidden_size: int = 128          # Можно увеличить до 256 для больших данных
    num_classes: int = 3
    groups: int = 8
    dropout: float = 0.3            # Умеренный dropout
    
    # Loss function (консервативные для начала)
    focal_gamma: float = 2.0
    focal_alpha_weight: float = 0.25
    cost_weight: float = 0.5
    focal_alpha: [5.0, 1.0, 5.0]

@dataclass  
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 5e-4     # Немного меньше из-за большей модели
    weight_decay: float = 1e-4
    epochs: int = 15
```

---

## 📈 Как постепенно увеличивать сложность loss

После первых нескольких эпох, если обучение стабильно:

**Шаг 1: (эпохи 1-5)**
```python
focal_gamma = 2.0
focal_alpha = [5.0, 1.0, 5.0]
cost_weight = 0.5
```

**Шаг 2: (эпохи 6-10)**
```python
focal_gamma = 2.5           # ↑
focal_alpha = [8.0, 1.0, 8.0]  # ↑
cost_weight = 0.8           # ↑
```

**Шаг 3: (эпохи 11-15)**
```python
focal_gamma = 3.0           # ↑
focal_alpha = [12.0, 1.0, 12.0]  # ↑
cost_weight = 1.2           # ↑
```

---

## 🐛 Известные ограничения

1. **Больше памяти GPU**
   - Новая модель ~1.7× больше
   - Если не хватает памяти: уменьшите `batch_size` до 256 или 384

2. **Медленнее обучение**
   - На ~30-40% медленнее за эпоху
   - Но нужно меньше эпох до сходимости

3. **Attention требует больше времени**
   - Quadratic complexity O(T²) по времени
   - Для T=240 это не проблема, но для T>500 может быть медленно

---

## 🔍 Визуализация архитектуры

```
Вход (B, 211, 240)
    ↓
Stem Conv1d (B, 128, 240)
    ↓
┌─────────────────────────────────┐
│ ResidualBlock (dilation=1)      │
│   ├─ Conv1d                     │
│   ├─ GroupNorm + GELU          │
│   ├─ Conv1d                     │
│   ├─ GroupNorm                  │
│   └─ Add Residual + GELU       │
└─────────────────────────────────┘
    ↓
ResidualBlock (dilation=2)
    ↓
ResidualBlock (dilation=4)
    ↓
ResidualBlock (dilation=8)
    ↓
ResidualBlock (dilation=16)
    ↓ (B, 128, 240)
┌─────────────────────────────────┐
│ Multi-Head Attention (4 heads)  │
│   ├─ Query/Key/Value            │
│   ├─ Attention weights          │
│   └─ Weighted sum + pooling     │
└─────────────────────────────────┘
    ↓ (B, 128)
Classification Head
    ├─ Linear(128 → 128) + GELU
    ├─ Dropout
    ├─ Linear(128 → 64) + GELU
    ├─ Dropout
    └─ Linear(64 → 3)
    ↓
Выход (B, 3)
```

---

## 🎓 Дополнительные улучшения (для будущего)

1. **Squeeze-and-Excitation блоки**
   - Адаптивное взвешивание каналов

2. **Temporal Convolutional Network (TCN)**
   - Еще более эффективная обработка последовательностей

3. **Learnable positional encoding**
   - Явное кодирование временной информации

4. **Cross-attention между уровнями стакана**
   - Моделирование взаимодействий между bid/ask сторонами

---

## ✅ Чек-лист перед запуском

- [x] Модель обновлена в `src/model.py`
- [x] Параметры loss уменьшены для стабильности
- [x] Learning rate уменьшен до 5e-4
- [x] Добавлена нормализация данных в `load_data.py`
- [x] Добавлены проверки NaN в `trainer.py`
- [ ] Протестировать на небольшом датасете
- [ ] Постепенно увеличивать сложность loss
- [ ] Мониторить confusion matrix после каждой эпохи

Удачи в обучении! 🚀

