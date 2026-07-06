import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    mean_squared_error, 
    mean_absolute_error
)
from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Classe base astratta (Template) per l'addestramento dei modelli.
    Gestisce il loop delle epoche, l'early stopping e lo step dello scheduler.
    """
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)
        
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for data, labels in tqdm(dataloader, desc="Training", leave=False):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        return running_loss / len(dataloader.dataset)

    @abstractmethod
    def evaluate(self, dataloader, test_mode=False):
        """
        Deve ritornare una tupla: (report/info, dict_risultati)
        dove dict_risultati contiene almeno la chiave "losses".
        """
        raise NotImplementedError("Il metodo evaluate deve restituire una tupla (report, dict_risultati) con almeno la chiave 'losses'.")

    def step_scheduler(self, val_loss):
        """Gestisce in modo sicuro vari tipi di scheduler."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def fit(self, train_loader, val_loader, epochs, patience=5, tolerance=1e-3):
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_loss = self.train_epoch(train_loader)

            _, val_data = self.evaluate(val_loader, test_mode=False)
            
            val_losses = val_data.get("losses", [])
            val_loss = np.mean(val_losses) if val_losses else 0.0
            
            # Estraiamo le metriche ignorando la lista delle losses
            metrics_only = {k: v for k, v in val_data.items() if k != "losses"}

            self.step_scheduler(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(metrics_only)

            # Stampa dinamica delle metriche
            metrics_str = " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in metrics_only.items()])
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {metrics_str}")

            # Early Stopping Logic
            if val_loss + tolerance < best_val_loss:
                print(f"(!) Nuovo miglior modello (Val Loss diminuita: {best_val_loss:.4f} --> {val_loss:.4f})")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")  # Salva i pesi del miglior modello
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Val Loss non migliorata. Patience: {patience_counter}/{patience} (Tol: {tolerance})")
                if patience_counter >= patience:
                    print("Early stopping attivato. Interruzione dell'addestramento.")
                    break

        print("\nCaricamento dei pesi del modello migliore...")
        self.model.load_state_dict(torch.load("best_model.pth"))
        os.remove("best_model.pth")  # Rimuove il file temporaneo dopo il caricamento


class ClassificationTrainer(BaseTrainer):
    """Trainer specifico per compiti di classificazione multi-classe."""
    
    def evaluate(self, dataloader, test_mode=False):
        self.model.eval()
        predictions, gts, losses = [], [], []

        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                data = data.to(self.device, non_blocking=True)
                labels_device = labels.to(self.device, non_blocking=True)

                outputs = self.model(data)

                if not test_mode:
                    loss = self.criterion(outputs, labels_device)
                    losses.append(loss.item())

                # Logica specifica classificazione multi-classe
                predicted = torch.argmax(outputs, 1)
                
                predictions.append(predicted.cpu().numpy())
                gts.append(labels.cpu().numpy())

        all_gts = np.hstack(gts)
        all_preds = np.hstack(predictions)
        
        report = classification_report(all_gts, all_preds, zero_division=0)
        accuracy = accuracy_score(all_gts, all_preds)

        return report, {"losses": losses, "accuracy": accuracy}


class RegressionTrainer(BaseTrainer):
    """Trainer specifico per compiti di regressione continua."""
    
    def evaluate(self, dataloader, test_mode=False):
        self.model.eval()
        predictions, gts, losses = [], [], []

        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                data = data.to(self.device, non_blocking=True)
                labels_device = labels.to(self.device, dtype=torch.float32, non_blocking=True)

                outputs = self.model(data)

                if not test_mode:
                    # Assumiamo output di forma [batch_size, 1] e label [batch_size, 1]
                    loss = self.criterion(outputs.squeeze(), labels_device.squeeze())
                    losses.append(loss.item())

                # Logica specifica regressione (nessun argmax)
                predictions.append(outputs.squeeze().cpu().numpy())
                gts.append(labels.cpu().numpy())

        all_gts = np.concatenate(gts).flatten()
        all_preds = np.concatenate(predictions).flatten()
        
        mse = mean_squared_error(all_gts, all_preds)
        mae = mean_absolute_error(all_gts, all_preds)

        return None, {"losses": losses, "mse": mse, "mae": mae}


class TrainerFactory:
    """Factory per l'istanziazione del trainer corretto."""
    
    @staticmethod
    def get_trainer(task_type, model, optimizer, criterion, device, scheduler=None):
        task_type = task_type.lower().strip()
        
        if task_type in ["classification", "multiclass"]:
            return ClassificationTrainer(model, optimizer, criterion, device, scheduler)
        elif task_type in ["regression"]:
            return RegressionTrainer(model, optimizer, criterion, device, scheduler)
        else:
            raise ValueError(f"Task type '{task_type}' non è supportato. Usa 'classification' o 'regression'.")