"""
Кастомная функция потерь с учетом стоимости ошибок и дисбаланса классов
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveFocalLoss(nn.Module):
    """
    Комбинированная функция потерь:
    1. Cost-Sensitive: учитывает различную стоимость ошибок
    2. Focal Loss: борется с дисбалансом классов
    
    Матрица стоимости (cost_matrix):
    - Строки: истинные классы (0=down, 1=flat, 2=up)
    - Столбцы: предсказанные классы
    
    Ваша специфика:
    - Предсказан down/up, истина противоположная: стоимость = 1.0 (максимальная ошибка)
    - Предсказан flat, истина up/down: стоимость = 0.5 (средняя ошибка)
    - Предсказан down/up, истина flat: стоимость = 0.75 (выше средней)
    - Правильное предсказание: стоимость = 0.0
    """
    
    def __init__(self, 
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 cost_weight: float = 1.0):
        """
        Args:
            gamma: параметр фокусировки для Focal Loss (обычно 2.0)
                   чем больше, тем сильнее фокус на сложных примерах
            alpha: вес для балансировки классов (0.25-0.75)
            cost_weight: вес для cost-sensitive компоненты (0.0-1.0)
                        0.0 = только Focal Loss
                        1.0 = равный баланс между Focal и Cost
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.cost_weight = cost_weight
        
        # Матрица стоимости ошибок
        # Строки: истинный класс [down=0, flat=1, up=2]
        # Столбцы: предсказанный класс [down=0, flat=1, up=2]
        cost_matrix = torch.tensor([
            [0.0,  0.75, 1.0],   # Истина: down
            [0.5,  0.0,  0.5],   # Истина: flat
            [1.0,  0.75, 0.0]    # Истина: up
        ], dtype=torch.float32)
        
        self.register_buffer('cost_matrix', cost_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: выход модели (batch_size, num_classes)
            targets: истинные метки (batch_size,)
        
        Returns:
            скалярное значение loss
        """
        # Вычисление вероятностей
        probs = F.softmax(logits, dim=1)
        
        # 1. Focal Loss компонента
        # Вероятность правильного класса
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # p_t для правильного класса
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Focal loss = alpha * focal_weight * ce_loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # 2. Cost-Sensitive компонента
        # Получаем стоимости для всех предсказаний
        # Убедимся, что cost_matrix на том же устройстве, что и targets
        cost_matrix = self.cost_matrix.to(targets.device)
        batch_costs = cost_matrix[targets]  # (batch_size, num_classes)
        
        # Взвешиваем предсказания по стоимости ошибок
        cost_weighted_probs = probs * batch_costs  # (batch_size, num_classes)
        cost_loss = cost_weighted_probs.sum(dim=1)  # (batch_size,)
        
        # 3. Комбинирование двух компонент
        combined_loss = focal_loss + self.cost_weight * cost_loss
        
        return combined_loss.mean()
    
    def get_cost_matrix(self) -> torch.Tensor:
        """Получить матрицу стоимости для анализа"""
        return self.cost_matrix.cpu()


class AdaptiveCostSensitiveFocalLoss(nn.Module):
    """
    Улучшенная версия с адаптивным весом alpha для каждого класса
    на основе частоты встречаемости классов
    """
    
    def __init__(self, 
                 class_frequencies: torch.Tensor,
                 gamma: float = 2.0,
                 cost_weight: float = 1.0):
        """
        Args:
            class_frequencies: частота встречаемости каждого класса
                              например: [0.1, 0.7, 0.2] для [down, flat, up]
            gamma: параметр фокусировки для Focal Loss
            cost_weight: вес для cost-sensitive компоненты
        """
        super().__init__()
        self.gamma = gamma
        self.cost_weight = cost_weight
        
        # Вычисление адаптивных весов alpha
        # Инверсная частота с нормализацией
        total = class_frequencies.sum()
        inv_freq = total / (class_frequencies + 1e-6)
        alpha = inv_freq / inv_freq.sum()
        
        self.register_buffer('alpha', alpha)
        
        # Матрица стоимости (та же самая)
        cost_matrix = torch.tensor([
            [0.0,  0.75, 1.0],
            [0.5,  0.0,  0.5],
            [1.0,  0.75, 0.0]
        ], dtype=torch.float32)
        
        self.register_buffer('cost_matrix', cost_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        
        # Focal Loss с адаптивным alpha
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        # Адаптивный alpha для каждого примера
        alpha = self.alpha.to(logits.device)
        alpha_t = (alpha.unsqueeze(0) * targets_one_hot).sum(dim=1)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Cost-Sensitive компонента
        cost_matrix = self.cost_matrix.to(targets.device)
        batch_costs = cost_matrix[targets]
        cost_weighted_probs = probs * batch_costs
        cost_loss = cost_weighted_probs.sum(dim=1)
        
        # Комбинирование
        combined_loss = focal_loss + self.cost_weight * cost_loss
        
        return combined_loss.mean()


def create_loss_function(class_distribution: dict = None,
                        gamma: float = 2.0,
                        alpha: float = 0.25,
                        cost_weight: float = 1.0,
                        adaptive: bool = True) -> nn.Module:
    """
    Создать функцию потерь
    
    Args:
        class_distribution: словарь с распределением классов
                           например: {0: 1000, 1: 7000, 2: 2000}
                           для классов [down, flat, up]
        gamma: параметр Focal Loss (рекомендуется 2.0-3.0)
        alpha: базовый вес для Focal Loss (если adaptive=False)
        cost_weight: вес cost-sensitive компоненты (0.5-1.5 рекомендуется)
        adaptive: использовать адаптивные веса на основе распределения
    
    Returns:
        Функция потерь
    """
    if adaptive and class_distribution is not None:
        # Вычисление частот классов
        total = sum(class_distribution.values())
        class_freq = torch.tensor([
            class_distribution.get(i, 1) / total 
            for i in range(3)
        ])
        
        print("=" * 70)
        print("ADAPTIVE COST-SENSITIVE FOCAL LOSS")
        print("=" * 70)
        print(f"Class distribution:")
        print(f"  Down (0): {class_distribution.get(0, 0):,} ({class_freq[0]:.2%})")
        print(f"  Flat (1): {class_distribution.get(1, 0):,} ({class_freq[1]:.2%})")
        print(f"  Up   (2): {class_distribution.get(2, 0):,} ({class_freq[2]:.2%})")
        
        criterion = AdaptiveCostSensitiveFocalLoss(
            class_frequencies=class_freq,
            gamma=gamma,
            cost_weight=cost_weight
        )
        
        print(f"\nAdaptive alpha weights: {criterion.alpha.cpu().numpy()}")
    else:
        print("=" * 70)
        print("COST-SENSITIVE FOCAL LOSS")
        print("=" * 70)
        
        criterion = CostSensitiveFocalLoss(
            gamma=gamma,
            alpha=alpha,
            cost_weight=cost_weight
        )
    
    print(f"\nParameters:")
    print(f"  Gamma: {gamma}")
    print(f"  Cost weight: {cost_weight}")
    print(f"\nCost Matrix:")
    print(criterion.get_cost_matrix())
    print("=" * 70)
    
    return criterion


# Пример использования
if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Пример 1: Простая версия
    print("1. Simple Cost-Sensitive Focal Loss")
    print("-" * 70)
    
    criterion = CostSensitiveFocalLoss(
        gamma=2.0,
        alpha=0.25,
        cost_weight=1.0
    )
    
    # Тестовые данные
    batch_size = 8
    num_classes = 3
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 1, 1, 0, 2, 1])  # down, flat, up, ...
    
    loss = criterion(logits, targets)
    print(f"\nTest loss: {loss.item():.4f}")
    print(f"Cost matrix:\n{criterion.get_cost_matrix()}\n")
    
    # Пример 2: Адаптивная версия с реальным распределением
    print("\n2. Adaptive Cost-Sensitive Focal Loss")
    print("-" * 70)
    
    # Симуляция дисбаланса: много flat, мало down/up
    class_distribution = {
        0: 1000,   # down - 10%
        1: 7000,   # flat - 70%
        2: 2000    # up   - 20%
    }
    
    criterion_adaptive = create_loss_function(
        class_distribution=class_distribution,
        gamma=2.5,
        cost_weight=1.2,
        adaptive=True
    )
    
    loss_adaptive = criterion_adaptive(logits, targets)
    print(f"\nTest loss: {loss_adaptive.item():.4f}")
    
    # Пример 3: Анализ различных ошибок
    print("\n3. Error Cost Analysis")
    print("-" * 70)
    
    # Создаем примеры различных ошибок
    test_cases = [
        (torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([0]), "Correct: down -> down"),
        (torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([1]), "down -> flat (cost 0.75)"),
        (torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([2]), "down -> up (cost 1.0)"),
        (torch.tensor([[0.0, 10.0, 0.0]]), torch.tensor([0]), "flat -> down (cost 0.5)"),
        (torch.tensor([[0.0, 10.0, 0.0]]), torch.tensor([2]), "flat -> up (cost 0.5)"),
    ]
    
    criterion_test = CostSensitiveFocalLoss(gamma=2.0, alpha=0.25, cost_weight=1.0)
    
    print("\nPrediction -> True Label (Expected Cost) | Loss Value")
    print("-" * 70)
    for logits_test, targets_test, description in test_cases:
        loss_test = criterion_test(logits_test, targets_test)
        print(f"{description:40s} | {loss_test.item():.4f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Рекомендации по настройке гиперпараметров:

1. gamma (Focal Loss):
   - 2.0-2.5: стандартное значение
   - 3.0-4.0: если очень сильный дисбаланс (>10:1)
   - 1.0-1.5: если дисбаланс небольшой

2. cost_weight:
   - 0.5-1.0: если дисбаланс не критичный
   - 1.0-1.5: если важны и баланс, и стоимость ошибок
   - 1.5-2.0: если стоимость ошибок критична

3. adaptive:
   - True: всегда рекомендуется при дисбалансе
   - False: только если классы сбалансированы

Для вашего случая рекомендую:
- gamma = 2.5 (умеренный фокус на сложных примерах)
- cost_weight = 1.0-1.2 (баланс между Focal и Cost)
- adaptive = True (автоматическая подстройка под дисбаланс)
    """)