"""
Модель DeepLOB для предсказания движения цен
"""
import torch
import torch.nn as nn
from typing import Optional
from focal_loss import FocalLoss
from cost_sensitive_loss import CostSensitiveLoss, get_default_cost_matrix
from cost_sensitive_focal_loss import CostSensitiveFocalLoss

class TemporalConvBlock(nn.Module):
    """Временной сверточный блок с групповой нормализацией"""
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, d: int = 1, groups: int = 8):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2*2)
        self.act2 = nn.GELU()
        g = min(groups, out_ch)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.act2(y)
        y = self.norm(y)
        return y[..., :x.shape[-1]]


class DeepLOBLike(nn.Module):
    """Модель DeepLOB для анализа стакана заявок"""
    
    def __init__(self, 
                 input_features: int, 
                 num_classes: int = 3, 
                 hidden_size: int = 128, 
                 groups: int = 8, 
                 dropout: float = 0.3):
        super().__init__()

        print(f"input_features :{input_features}")
        
        self.stem = nn.Conv1d(input_features, hidden_size, kernel_size=1)
        
        # Временные сверточные блоки с разными дилатациями
        self.b1 = TemporalConvBlock(hidden_size, hidden_size, k=5, d=1, groups=groups)
        self.b2 = TemporalConvBlock(hidden_size, hidden_size, k=5, d=2, groups=groups)
        self.b3 = TemporalConvBlock(hidden_size, hidden_size, k=5, d=4, groups=groups)
        
        # Глобальное усреднение и классификатор
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы (batch_size, features, time_steps)
        Returns:
            Логиты классификации формы (batch_size, num_classes)
        """
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

def create_model(config) -> tuple[nn.Module, nn.Module]:
    """
    Создать модель и функцию потерь
    
    Args:
        config: Конфигурация модели
        
    Returns:
        Кортеж (модель, функция потерь)
    """
    model = DeepLOBLike(
        input_features=getattr(config, 'input_features', 211),  # Количество признаков из конфига
        num_classes=config.num_classes,
        hidden_size=config.hidden_size,
        groups=config.groups,
        dropout=config.dropout
    )
    
    # Создание функции потерь
    if hasattr(config, 'use_cost_sensitive_focal') and config.use_cost_sensitive_focal:
        # Комбинированный Cost-Sensitive Focal Loss (по умолчанию)
        criterion = CostSensitiveFocalLoss(
            gamma=getattr(config, 'focal_gamma', 2.0),
            alpha=getattr(config, 'focal_alpha_weight', 0.25),
            cost_weight=getattr(config, 'cost_weight', 1.0)
        )
    elif hasattr(config, 'use_cost_sensitive') and config.use_cost_sensitive:
        # Только стоимостно-чувствительный лосс
        cost_matrix = get_default_cost_matrix()
        criterion = CostSensitiveLoss(cost_matrix)
    else:
        # Только Focal Loss
        criterion = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
    
    return model, criterion


def create_optimizer_and_scheduler(model: nn.Module, config):
    """
    Создать оптимизатор и планировщик обучения
    
    Args:
        model: Модель для оптимизации
        config: Конфигурация обучения
        
    Returns:
        Кортеж (оптимизатор, планировщик, scaler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.scheduler_t_max, 
            eta_min=config.scheduler_eta_min
        )
    else:
        scheduler = None
    
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))
    
    return optimizer, scheduler, scaler
