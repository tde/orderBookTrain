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
    
    def _print_training_info(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Вывести информацию о модели и обучении"""
        print("\n" + "="*70)
        print("🚀 ИНФОРМАЦИЯ ОБ ОБУЧЕНИИ")
        print("="*70)
        
        # Информация о модели
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n📊 МОДЕЛЬ:")
        print(f"   Всего параметров: {total_params:,}")
        print(f"   Обучаемых параметров: {trainable_params:,}")
        print(f"   Размер модели: ~{total_params * 4 / 1024 / 1024:.2f} МБ (float32)")
        
        # Информация о данных
        print("\n📦 ДАННЫЕ:")
        print(f"   Train батчей: {len(train_loader)}")
        print(f"   Val батчей: {len(val_loader)}")
        
        # Проверка первого батча для размерностей
        for xb, yb in train_loader:
            print(f"   Размер батча: {xb.shape}")
            print(f"   Тип данных: {xb.dtype}")
            print(f"   Устройство: {xb.device}")
            unique_labels, counts = torch.unique(yb, return_counts=True)
            print(f"   Классы в батче: {unique_labels.tolist()}")
            print(f"   Распределение: {counts.tolist()}")
            break
        
        # Информация об оптимизации
        print("\n⚙️  ОПТИМИЗАТОР:")
        print(f"   Тип: {type(self.optimizer).__name__}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"   Weight decay: {self.optimizer.param_groups[0]['weight_decay']:.2e}")
        if self.scheduler is not None:
            print(f"   Scheduler: {type(self.scheduler).__name__}")
        print(f"   Gradient clipping: {self.grad_clip_norm}")
        
        # Информация о loss
        print("\n📉 LOSS FUNCTION:")
        print(f"   Тип: {type(self.criterion).__name__}")
        if hasattr(self.criterion, 'gamma'):
            print(f"   focal_gamma: {self.criterion.gamma}")
        if hasattr(self.criterion, 'alpha'):
            print(f"   focal_alpha_weight: {self.criterion.alpha}")
        if hasattr(self.criterion, 'cost_weight'):
            print(f"   cost_weight: {self.criterion.cost_weight}")
        if hasattr(self.criterion, 'label_smoothing'):
            print(f"   label_smoothing: {self.criterion.label_smoothing}")
        
        # Информация об обучении
        print("\n🎯 ОБУЧЕНИЕ:")
        print(f"   Эпох: {epochs}")
        print(f"   Устройство: {self.device}")
        print(f"   Mixed precision: {self.scaler is not None and self.scaler.is_enabled()}")
        
        print("\n" + "="*70)
        print("▶️  НАЧАЛО ОБУЧЕНИЯ...")
        print("="*70 + "\n")
    
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
            
            # Проверка на NaN/Inf в loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  WARNING: Loss is {loss.item()}, skipping batch")
                continue
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Проверка градиентов на NaN
                grad_norm = U.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"⚠️  WARNING: Gradient norm is {grad_norm}, skipping batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = U.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"⚠️  WARNING: Gradient norm is {grad_norm}, skipping batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total_samples += xb.size(0)
        
        train_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        train_acc = correct / total_samples if total_samples > 0 else 0.0
        
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
        # Вывод информации о модели перед началом обучения
        if verbose:
            self._print_training_info(train_loader, val_loader, epochs)
        
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
