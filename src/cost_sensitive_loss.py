"""
Стоимостно-чувствительный лосс для финансовых данных
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveLoss(nn.Module):
    """
    Стоимостно-чувствительный лосс для финансовых данных
    
    Матрица C: shape (num_classes, num_classes)
    C[y, k] — стоимость предсказать k, когда истинный класс y.
    """
    
    def __init__(self, cost_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("C", cost_matrix.float())
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Логиты модели (batch_size, num_classes)
            targets: Истинные метки (batch_size,)
        Returns:
            Стоимостно-чувствительная потеря
        """
        # probs: softmax по классам
        probs = F.softmax(logits, dim=1)  # (B, K)
        # выбираем строку C для каждого таргета y
        C_y = self.C[targets]  # (B, K)
        # ожидаемая стоимость = сумма p_k * C[y,k]
        loss = (probs * C_y).sum(dim=1)  # (B,)
        return loss.mean()


def get_default_cost_matrix() -> torch.Tensor:
    """
    Возвращает матрицу стоимости по умолчанию для финансовых данных
    
    Returns:
        Матрица стоимости формы (3, 3) для классов [down, flat, up]
    """
    return torch.tensor([
        [0.0, 0.40, 1.00],   # true=down: ошибка "up" = 1.00, "flat" = 0.40
        [0.73, 0.00, 0.73],  # true=flat: любая "up"/"down" = 0.73 (дорого)
        [1.0, 0.40, 0.00]    # true=up:  ошибка "down" = 1.00, "flat" = 0.40
    ])