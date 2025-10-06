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


class LazyWindowDataset(Dataset):
    """
    Dataset создающий окна "на лету" без предварительного копирования всех данных.
    Экономит память при работе с большими датасетами.
    """
    
    def __init__(self, X2D: np.ndarray, y_filtered: np.ndarray, window_length: int):
        """
        Args:
            X2D: 2D массив признаков формы (N, F) - исходные данные
            y_filtered: Массив меток формы (N,) - метки для каждой строки
            window_length: Длина окна T
        """
        if X2D.shape[0] < window_length:
            raise ValueError(f"Мало данных для окна: N={X2D.shape[0]} < T={window_length}")
        
        self.X2D = X2D.astype(np.float32, copy=False)
        self.y_filtered = y_filtered.astype(np.int64, copy=False)
        self.T = window_length
        self.N, self.F = X2D.shape
        
    def __len__(self) -> int:
        return self.N - self.T + 1
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Создает окно на лету для индекса i"""
        # Берем окно [i : i+T]
        window = self.X2D[i : i + self.T]  # (T, F)
        # Транспонируем для Conv1d: (F, T)
        x = torch.from_numpy(window.T.copy())
        # Метка соответствует последнему элементу окна
        y = torch.tensor(self.y_filtered[i + self.T - 1])
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
        X_train, X_val, X_test: Массивы признаков (уже оконные данные формы (N, T, F))
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


def create_lazy_data_loaders(X2D_train: np.ndarray, 
                             y_train: np.ndarray,
                             X2D_val: np.ndarray, 
                             y_val: np.ndarray,
                             X2D_test: np.ndarray, 
                             y_test: np.ndarray,
                             window_length: int,
                             batch_size: int = 512,
                             num_classes: int = 3,
                             seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создать память-эффективные DataLoader'ы с lazy окнами
    
    Args:
        X2D_train, X2D_val, X2D_test: 2D массивы признаков формы (N, F)
        y_train, y_val, y_test: Массивы меток формы (N,)
        window_length: Длина окна T
        batch_size: Размер батча
        num_classes: Количество классов
        seed: Семя для воспроизводимости
        
    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    # Создаем LazyWindowDataset'ы
    train_ds = LazyWindowDataset(X2D_train, y_train, window_length)
    val_ds = LazyWindowDataset(X2D_val, y_val, window_length)
    test_ds = LazyWindowDataset(X2D_test, y_test, window_length)
    
    # Для lazy dataset нужно создать метки для окон
    # Метка окна = метка последнего элемента
    y_train_windowed = y_train[window_length-1:]
    
    # Создаем сбалансированный sampler для обучения
    base_sampler = BalancedBatchSampler(
        y_train_windowed, 
        batch_size=batch_size, 
        num_classes=num_classes, 
        seed=seed
    )
    batch_sampler = BalancedBatchBatchSampler(base_sampler, batch_size)
    
    # Создаем DataLoader'ы
    train_dl = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_dl, val_dl, test_dl
