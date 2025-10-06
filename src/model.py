"""
Модель DeepLOB для предсказания движения цен
Улучшенная версия с Residual connections и Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from focal_loss import FocalLoss
from cost_sensitive_loss import CostSensitiveLoss, get_default_cost_matrix
from cost_sensitive_focal_loss import CostSensitiveFocalLoss


class TemporalConvBlockWithResidual(nn.Module):
    """
    Временной сверточный блок с residual connection
    """
    
    def __init__(self, channels: int, k: int = 5, dilation: int = 1, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, dilation=dilation, padding=pad)
        
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(num_groups=g, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=g, num_channels=channels)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = out[..., :x.shape[-1]]  # Crop to original length
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[..., :x.shape[-1]]  # Crop to original length
        out = self.norm2(out)
        
        # Residual connection
        out = out + identity
        out = self.act(out)
        
        return out


class TemporalAttention(nn.Module):
    """
    Attention механизм для временных последовательностей
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size должен делиться на num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)
        Returns:
            (batch_size, channels)
        """
        B, C, T = x.shape
        
        # Transpose to (batch, time, channels)
        x = x.transpose(1, 2)  # (B, T, C)
        
        # Multi-head attention
        Q = self.query(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = self.key(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ V  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        out = self.out(out)
        
        # Global average pooling over time
        out = out.mean(dim=1)  # (B, C)
        
        return out


class DeepLOBLike(nn.Module):
    """
    Улучшенная модель DeepLOB с:
    - Residual connections
    - Больше дилатаций (1, 2, 4, 8, 16)
    - Attention механизм вместо pooling
    """
    
    def __init__(self, 
                 input_features: int, 
                 num_classes: int = 3, 
                 hidden_size: int = 128, 
                 groups: int = 8, 
                 dropout: float = 0.3):
        super().__init__()
        
        print(f"🔧 Создание улучшенной модели:")
        print(f"   input_features: {input_features}")
        print(f"   hidden_size: {hidden_size}")
        print(f"   num_classes: {num_classes}")
        print(f"   Residual connections: ✓")
        print(f"   Dilations: [1, 2, 4, 8, 16]")
        print(f"   Attention mechanism: ✓")
        
        # Stem: проецируем входные признаки в hidden_size
        self.stem = nn.Conv1d(input_features, hidden_size, kernel_size=1)
        
        # Временные сверточные блоки с residual connections и разными дилатациями
        # Используем меньший dropout в residual блоках (0.1), основной dropout в голове
        block_dropout = 0.1
        self.blocks = nn.ModuleList([
            TemporalConvBlockWithResidual(hidden_size, k=5, dilation=1, groups=groups, dropout=block_dropout),
            TemporalConvBlockWithResidual(hidden_size, k=5, dilation=2, groups=groups, dropout=block_dropout),
            TemporalConvBlockWithResidual(hidden_size, k=5, dilation=4, groups=groups, dropout=block_dropout),
            TemporalConvBlockWithResidual(hidden_size, k=5, dilation=8, groups=groups, dropout=block_dropout),
            TemporalConvBlockWithResidual(hidden_size, k=5, dilation=16, groups=groups, dropout=block_dropout),
        ])
        
        # Attention механизм вместо AdaptiveAvgPool
        self.attention = TemporalAttention(hidden_size, num_heads=4)
        
        # Классификационная голова
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы (batch_size, features, time_steps)
        Returns:
            Логиты классификации формы (batch_size, num_classes)
        """
        # Проверка входных данных
        if x.dtype != torch.float32:
            x = x.float()
        
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (B, F, T), got shape {x.shape}")
        
        # Stem
        x = self.stem(x)
        
        # Последовательное применение residual блоков с разными дилатациями
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Проверка на NaN после каждого блока
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after block {i}")
        
        # Attention pooling
        x = self.attention(x)  # (B, hidden_size)
        
        if torch.isnan(x).any():
            raise ValueError(f"NaN detected after attention")
        
        # Classification head
        out = self.head(x)
        
        return out

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
