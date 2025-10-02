"""
Focal Loss для решения проблемы дисбаланса классов
"""
import torch
import torch.nn as nn
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss для решения проблемы дисбаланса классов"""
    
    def __init__(self, 
                 alpha: Optional[list] = None, 
                 gamma: float = 2.0, 
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = None if alpha is None else torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Логиты модели (batch_size, num_classes)
            targets: Истинные метки (batch_size,)
        Returns:
            Focal Loss
        """
        ce = nn.functional.cross_entropy(
            logits, targets, reduction='none',
            weight=(self.alpha.to(logits.device) if self.alpha is not None else None)
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()
