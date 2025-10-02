"""
Dataset и Sampler для обучения модели DeepLOB
"""
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple


class NpWindowDataset(Dataset):
    """Dataset для оконных данных из numpy массивов"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Массив признаков формы (N, T, F)
            y: Массив меток формы (N,)
        """
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[i].T.copy())  # (F, T) для Conv1d
        y = torch.tensor(self.y[i])
        return x, y


class BalancedBatchSampler(Sampler):
    """
    Формирует батчи одинакового размера из равного числа классов (по возможности).
    Например, при batch_size=512 и 3 классах -> по ~170 сэмплов с класса.
    Для редких классов крутит циклически (replacement=True).
    """
    
    def __init__(self, 
                 labels: np.ndarray, 
                 batch_size: int = 512, 
                 num_classes: int = 3, 
                 seed: int = 42):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rng = random.Random(seed)

        # Группируем индексы по классам
        self.idx_by_cls = [np.flatnonzero(self.labels == c).tolist() for c in range(num_classes)]
        for lst in self.idx_by_cls:
            self.rng.shuffle(lst)

        self.ptr = [0] * num_classes
        self.len_dataset = len(self.labels)
        # Сколько батчей сделаем за эпоху (ориентируемся на «средний» объём)
        self.batches_per_epoch = math.floor(self.len_dataset / self.batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        per_cls = [self.batch_size // self.num_classes] * self.num_classes
        rem = self.batch_size - sum(per_cls)
        # Раздадим остаток по первым классам
        for c in range(rem):
            per_cls[c] += 1

        for _ in range(self.batches_per_epoch):
            batch_idx = []
            for c in range(self.num_classes):
                need = per_cls[c]
                lst = self.idx_by_cls[c]
                # Если не хватает — крутим по кругу
                if self.ptr[c] + need > len(lst):
                    # Перетасуем чтобы не циклилось одинаково
                    self.rng.shuffle(lst)
                    self.ptr[c] = 0
                batch_idx.extend(lst[self.ptr[c]: self.ptr[c] + need])
                self.ptr[c] += need
            self.rng.shuffle(batch_idx)
            yield from batch_idx  # DataLoader сам нарежет по batch_size


class BalancedBatchBatchSampler(Sampler):
    """Обёртка: превращает поток индексов из BalancedBatchSampler в батчи по batch_size."""
    
    def __init__(self, base_sampler: BalancedBatchSampler, batch_size: int):
        self.base = base_sampler
        self.batch_size = batch_size
    
    def __len__(self) -> int:
        return len(self.base)
    
    def __iter__(self):
        it = iter(self.base)
        for _ in range(len(self.base)):
            batch = [next(it) for __ in range(self.batch_size)]
            yield batch


def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       batch_size: int = 512,
                       num_classes: int = 3,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создать DataLoader'ы для обучения, валидации и тестирования
    
    Args:
        X_train, X_val, X_test: Массивы признаков
        y_train, y_val, y_test: Массивы меток
        batch_size: Размер батча
        num_classes: Количество классов
        seed: Семя для воспроизводимости
        
    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    # Создаем Dataset'ы
    train_ds = NpWindowDataset(X_train, y_train)
    val_ds = NpWindowDataset(X_val, y_val)
    test_ds = NpWindowDataset(X_test, y_test)
    
    # Создаем сбалансированный sampler для обучения
    base_sampler = BalancedBatchSampler(
        y_train, 
        batch_size=batch_size, 
        num_classes=num_classes, 
        seed=seed
    )
    batch_sampler = BalancedBatchBatchSampler(base_sampler, batch_size)
    
    # Создаем DataLoader'ы
    train_dl = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl
