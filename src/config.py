"""
Конфигурация для обучения модели DeepLOB
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Конфигурация данных"""
    data_folder: str = "data"
    train_dates: list = None  # Список дней для обучения
    val_date: str = None      # День для валидации
    test_date: str = None     # День для тестирования
    instrument: str = "Si-12.25"
    
    def __post_init__(self):
        if self.train_dates is None:
            self.train_dates = ["2025-09-22"]
        if self.val_date is None:
            self.val_date = "2025-09-23"
        if self.test_date is None:
            self.test_date = "2025-09-24"
    
    def get_features_file(self, date: str) -> str:
        return os.path.join(self.data_folder, f"{self.instrument}_{date}_features.npy")
    
    def get_prices_file(self, date: str) -> str:
        return os.path.join(self.data_folder, f"{self.instrument}_{date}_prices.csv")
    
    @property
    def features_file(self) -> str:
        """Для обратной совместимости - возвращает первый день обучения"""
        return self.get_features_file(self.train_dates[0])
    
    @property
    def prices_file(self) -> str:
        """Для обратной совместимости - возвращает первый день обучения"""
        return self.get_prices_file(self.train_dates[0])


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Параметры задачи
    tick_size: float = 1.0
    theta_ticks: int = 5
    horizon_sec: float = 2.0
    window_length: int = 240
    
    # Архитектура модели (УМЕНЬШЕНА для борьбы с overfitting)
    input_features: int = 211        # Количество входных признаков
    hidden_size: int = 96            # УМЕНЬШЕНО с 128 (меньше параметров)
    num_classes: int = 3
    groups: int = 8
    dropout: float = 0.5             # УВЕЛИЧЕН dropout (сильная регуляризация)
    
    # Loss function (ПРОСТОЙ weighted CE - радикальное изменение!)
    use_cost_sensitive_focal: bool = False # ВЫКЛЮЧЕН - не помогает
    use_cost_sensitive: bool = False       # ВЫКЛЮЧЕН
    use_weighted_ce: bool = True           # НОВОЕ: простой weighted CrossEntropy
    focal_alpha: list = None
    focal_gamma: float = 2.5
    focal_alpha_weight: float = 0.25
    cost_weight: float = 1.0
    label_smoothing: float = 0.1           # СНИЖЕНО обратно (0.2 было слишком много)
    
    def __post_init__(self):
        if self.focal_alpha is None:
            # СБАЛАНСИРОВАННЫЕ веса (было 20 - слишком много!)
            self.focal_alpha = [10.0, 1.0, 10.0]


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    # Размеры батчей
    batch_size: int = 512
    
    # Оптимизатор (УМЕРЕННЫЙ LR)
    learning_rate: float = 1e-4           # УВЕЛИЧЕН (5e-5 было слишком мало)
    weight_decay: float = 5e-4            # СНИЖЕН (1e-3 было слишком много)
    
    # Scheduler
    scheduler_type: str = "cosine"
    scheduler_t_max: int = 10
    scheduler_eta_min: float = 3e-5
    
    # Обучение
    epochs: int = 15                  # УВЕЛИЧЕНО обратно (с низким LR нужно больше эпох)
    grad_clip_norm: float = 0.3       # ЕЩЕ МЕНЬШЕ (очень агрессивная стрижка)
    
    # Разделение данных
    train_split: float = 0.70
    val_split: float = 0.85
    
    # Случайность
    seed: int = 42
    
    # Устройство
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Общая конфигурация"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    def print_config(self):
        """Вывести все параметры конфигурации"""
        print("\n" + "="*70)
        print("📋 КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
        print("="*70)
        
        print("\n🗂️  ДАННЫЕ:")
        print(f"   Дни обучения: {self.data.train_dates}")
        print(f"   День валидации: {self.data.val_date}")
        print(f"   День тестирования: {self.data.test_date}")
        print(f"   Папка данных: {self.data.data_folder}")
        
        print("\n🏗️  АРХИТЕКТУРА МОДЕЛИ:")
        print(f"   input_features: {self.model.input_features}")
        print(f"   hidden_size: {self.model.hidden_size}")
        print(f"   num_classes: {self.model.num_classes}")
        print(f"   groups: {self.model.groups}")
        print(f"   dropout: {self.model.dropout}")
        print(f"   window_length: {self.model.window_length}")
        
        print("\n📊 LOSS FUNCTION:")
        print(f"   use_cost_sensitive_focal: {self.model.use_cost_sensitive_focal}")
        print(f"   focal_alpha: {self.model.focal_alpha}")
        print(f"   focal_gamma: {self.model.focal_gamma}")
        print(f"   focal_alpha_weight: {self.model.focal_alpha_weight}")
        print(f"   cost_weight: {self.model.cost_weight}")
        if hasattr(self.model, 'label_smoothing'):
            print(f"   label_smoothing: {self.model.label_smoothing}")
        
        print("\n⚙️  ОПТИМИЗАЦИЯ:")
        print(f"   batch_size: {self.training.batch_size}")
        print(f"   learning_rate: {self.training.learning_rate}")
        print(f"   weight_decay: {self.training.weight_decay}")
        print(f"   epochs: {self.training.epochs}")
        print(f"   grad_clip_norm: {self.training.grad_clip_norm}")
        print(f"   scheduler_type: {self.training.scheduler_type}")
        print(f"   device: {self.training.device}")
        
        print("\n" + "="*70 + "\n")
    
    @classmethod
    def default(cls):
        """Создать конфигурацию по умолчанию"""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig()
        )
    
    @classmethod
    def from_args(cls, args):
        """Создать конфигурацию из аргументов командной строки"""
        # Обработка списка дней для обучения
        train_dates = getattr(args, 'train_dates', None)
        if train_dates is None:
            # Если train_dates не задан, используем train_date для обратной совместимости
            train_date = getattr(args, 'train_date', '2025-09-22')
            train_dates = [train_date]
        elif isinstance(train_dates, str):
            # Если передан как строка, разделяем по запятым
            train_dates = [d.strip() for d in train_dates.split(',')]
        
        data_config = DataConfig(
            train_dates=train_dates,
            val_date=getattr(args, 'val_date', '2025-09-23'),
            test_date=getattr(args, 'test_date', '2025-09-24'),
            data_folder=getattr(args, 'data_folder', 'data'),
            instrument=getattr(args, 'instrument', 'Si-12.25')
        )
        
        # Определяем какую функцию потерь использовать
        use_cost_sensitive_focal = getattr(args, 'use_cost_sensitive_focal', True)
        use_cost_sensitive = getattr(args, 'use_cost_sensitive', False)
        use_focal_loss = getattr(args, 'use_focal_loss', False)
        
        # Приоритет: cost_sensitive_focal > cost_sensitive > focal_only
        if use_focal_loss:
            use_cost_sensitive_focal = False
            use_cost_sensitive = False
        elif use_cost_sensitive:
            use_cost_sensitive_focal = False
        
        model_config = ModelConfig(
            tick_size=getattr(args, 'tick_size', 1.0),
            theta_ticks=getattr(args, 'theta_ticks', 5),
            horizon_sec=getattr(args, 'horizon_sec', 2.0),
            window_length=getattr(args, 'window_length', 240),
            hidden_size=getattr(args, 'hidden_size', 128),
            dropout=getattr(args, 'dropout', 0.3),
            use_cost_sensitive_focal=use_cost_sensitive_focal,
            use_cost_sensitive=use_cost_sensitive,
            focal_gamma=getattr(args, 'focal_gamma', 2.0),
            focal_alpha_weight=getattr(args, 'focal_alpha_weight', 0.25),
            cost_weight=getattr(args, 'cost_weight', 1.0)
        )
        
        training_config = TrainingConfig(
            batch_size=getattr(args, 'batch_size', 512),
            learning_rate=getattr(args, 'learning_rate', 1e-3),
            epochs=getattr(args, 'epochs', 15),
            seed=getattr(args, 'seed', 42)
        )
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config
        )
