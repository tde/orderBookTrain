"""
Модуль для обучения модели DeepLOB
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.utils as U
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from typing import Tuple, Dict, Any


class Trainer:
    """Класс для обучения модели DeepLOB"""
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 scaler: torch.cuda.amp.GradScaler = None,
                 device: str = "cpu",
                 grad_clip_norm: float = 1.0):
        """
        Args:
            model: Модель для обучения
            criterion: Функция потерь
            optimizer: Оптимизатор
            scheduler: Планировщик обучения
            scaler: Scaler для смешанной точности
            device: Устройство для вычислений
            grad_clip_norm: Норма для обрезки градиентов
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        
        # Состояние обучения
        self.best_score = -1.0
        self.best_state = None
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                U.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                U.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total_samples += xb.size(0)
        
        train_loss = total_loss / total_samples
        train_acc = correct / total_samples
        
        return {
            'loss': train_loss,
            'accuracy': train_acc
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Оценка модели на валидационном наборе"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                
                # Мониторим «обычный» CE (он коррелирует со здравым смыслом)
                loss = nn.functional.cross_entropy(logits, yb, reduction='mean')
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)
                
                y_true.append(yb.cpu().numpy())
                y_pred.append(logits.argmax(1).cpu().numpy())
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        val_loss = total_loss / total_samples
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        
        # F1 для каждого класса
        f1_scores = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None)
        f1_down, f1_flat, f1_up = f1_scores
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        return {
            'loss': val_loss,
            'macro_f1': macro_f1,
            'f1_down': f1_down,
            'f1_flat': f1_flat,
            'f1_up': f1_up,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Полное обучение модели
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: Количество эпох
            verbose: Выводить ли прогресс
            
        Returns:
            Словарь с историей обучения и лучшим состоянием модели
        """
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Обновление learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Валидация
            val_metrics = self.evaluate(val_loader)
            
            # Сохранение истории
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Сохранение лучшей модели
            if val_metrics['macro_f1'] > self.best_score:
                self.best_score = val_metrics['macro_f1']
                self.best_state = {k: v.detach().cpu().clone() 
                                 for k, v in self.model.state_dict().items()}
            
            # Вывод прогресса
            if verbose:
                elapsed = time.time() - start_time
                print(f"[{epoch:02d}] "
                      f"train_loss={train_metrics['loss']:.4f} "
                      f"acc={train_metrics['accuracy']:.3f} | "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"macroF1={val_metrics['macro_f1']:.3f} "
                      f"F1↓={val_metrics['f1_down']:.3f} "
                      f"F1○={val_metrics['f1_flat']:.3f} "
                      f"F1↑={val_metrics['f1_up']:.3f} "
                      f"({elapsed:.1f}s)")
                
                # Показываем матрицу ошибок на определенных эпохах
                if epoch in (1, 5, 10, epochs):
                    print("Confusion matrix [rows=true, cols=pred]:")
                    print(val_metrics['confusion_matrix'])
                    print()
        
        return {
            'best_score': self.best_score,
            'best_state': self.best_state,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Тестирование модели
        
        Args:
            test_loader: DataLoader для тестирования
            
        Returns:
            Словарь с метриками тестирования
        """
        # Загружаем лучшее состояние
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        test_metrics = self.evaluate(test_loader)
        
        # Детальный отчет
        print(f"\nЛучший macro-F1 (val): {self.best_score:.4f}")
        print(classification_report(
            test_metrics['y_true'], 
            test_metrics['y_pred'], 
            digits=3, 
            target_names=["down", "flat", "up"]
        ))
        
        return test_metrics
    
    def save_model(self, path: str):
        """Сохранить лучшую модель"""
        if self.best_state is not None:
            torch.save(self.best_state, path)
            print(f"Модель сохранена: {path}")
        else:
            print("Нет сохраненного состояния модели")
    
    def load_model(self, path: str):
        """Загрузить модель из файла"""
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Модель загружена: {path}")
        else:
            print(f"Файл модели не найден: {path}")
    
    def train_sequential_days(self, 
                             train_data_loaders: list, 
                             val_loader: DataLoader,
                             epochs_per_day: int,
                             save_path_template: str = "model_day_{}.pt",
                             verbose: bool = True) -> Dict[str, Any]:
        """
        Последовательное обучение на нескольких днях
        
        Args:
            train_data_loaders: Список DataLoader'ов для каждого дня обучения
            val_loader: DataLoader для валидации
            epochs_per_day: Количество эпох на каждый день
            save_path_template: Шаблон пути для сохранения модели после каждого дня
            verbose: Выводить ли подробную информацию
            
        Returns:
            Словарь с результатами обучения
        """
        all_results = []
        
        for day_idx, train_loader in enumerate(train_data_loaders):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ОБУЧЕНИЕ НА ДНЕ {day_idx + 1}/{len(train_data_loaders)}")
                print(f"{'='*60}")
            
            # Обучение на текущем дне
            day_results = self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_day,
                verbose=verbose
            )
            
            all_results.append(day_results)
            
            # Сохранение модели после каждого дня
            save_path = save_path_template.format(day_idx + 1)
            self.save_model(save_path)
            
            if verbose:
                print(f"День {day_idx + 1} завершен. Лучший macro-F1: {day_results['best_score']:.4f}")
        
        # Возвращаем результаты последнего дня (финальное состояние модели)
        return all_results[-1] if all_results else None


def create_trainer(model: nn.Module, 
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                   scaler: torch.cuda.amp.GradScaler = None,
                   device: str = "cpu",
                   grad_clip_norm: float = 1.0) -> Trainer:
    """
    Создать объект Trainer
    
    Args:
        model: Модель для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        scheduler: Планировщик обучения
        scaler: Scaler для смешанной точности
        device: Устройство для вычислений
        grad_clip_norm: Норма для обрезки градиентов
        
    Returns:
        Объект Trainer
    """
    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        grad_clip_norm=grad_clip_norm
    )
