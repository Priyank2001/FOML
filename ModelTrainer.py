from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

class ModelTrainer(ABC):
    def __init__(self, model, optimizer, train_loader, val_loader, device, save_path):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        self.best_vloss = float('inf')

    def train(self, epochs):
        """The main training loop shared by all models."""
        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            # These methods are defined in the child classes
            train_metrics = self.train_epoch(epoch, epochs)
            val_metrics = self.validate_epoch(epoch, epochs)
            
            print(f"End of Epoch [{epoch+1}/{epochs}]")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics:   {val_metrics}")
            
            # Use the validation loss to determine if we should save the model
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_vloss:
                print(f"*** Validation loss improved ({self.best_vloss:.4f} --> {val_loss:.4f}). Saving model! ***")
                self.best_vloss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
            print("-" * 60)

    @abstractmethod
    def train_epoch(self, epoch, total_epochs):
        """Must return a dictionary of metrics, e.g., {'loss': 0.5}"""
        pass

    @abstractmethod
    def validate_epoch(self, epoch, total_epochs):
        """Must return a dictionary of metrics, e.g., {'loss': 0.4}"""
        pass





#################################################################################################################
### ------------------------------------- SNN TRAINER --------------------------------------------------------###
#################################################################################################################    
class SNNTrainer(ModelTrainer):
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device, save_path="best_snn.pth"):
        # Pass the common arguments up to the parent class
        super().__init__(model, optimizer, train_loader, val_loader, device, save_path)
        self.criterion = criterion 

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train SNN]", unit="batch")
        
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        return {
            'loss': running_loss / len(self.train_loader),
            'accuracy': 100 * correct / total
        }

    def validate_epoch(self, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        return {
            'loss': running_loss / len(self.val_loader),
            'accuracy': 100 * correct / total
        }
    
    def evaluate(self, test_loader):
        self.model.eval()
        print(f"Evaluating [SNN] on the Test Set...\n")

        test_loss = 0.0
        correct, total = 0, 0
        all_predictions, all_true_labels = [], []

        with torch.no_grad():
            for batch_X, batch_y in test_loader: 
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total

        print("="*45)
        print(" FINAL TEST METRICS: SNN ")
        print("="*45)
        print(f" Test Loss:     {avg_test_loss:.4f}")
        print(f" Test Accuracy: {test_accuracy:.2f}%")
        print("="*45)
        
        return avg_test_loss, test_accuracy, all_predictions, all_true_labels

class VAETrainer(ModelTrainer):
    # ... (Keep your existing __init__, train_epoch, and validate_epoch methods) ...

    def evaluate(self, test_loader):
        self.model.eval()
        print(f"Evaluating [VAE] on the Test Set...\n")

        test_loss = 0.0
        # Instead of predictions, we will save images to visualize the VAE's performance
        original_images = []
        reconstructed_images = []

        with torch.no_grad():
            for batch_X, _ in test_loader: # Ignore labels for VAE
                batch_X = batch_X.to(self.device)
                
                # Unpack the VAE output
                reconstructed_X, mu, log_var = self.model(batch_X)
                
                # Calculate custom VAE loss
                loss = self.vae_loss_fn(reconstructed_X, batch_X, mu, log_var)
          
                test_loss += loss.item()
                
                # Save just the very first batch of images for plotting later
                if len(original_images) == 0:
                    original_images.extend(batch_X.cpu().numpy())
                    reconstructed_images.extend(reconstructed_X.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)

        print("="*45)
        print(" FINAL TEST METRICS: VAE ")
        print("="*45)
        print(f" Test Loss (Reconstruction + KL): {avg_test_loss:.4f}")
        print("="*45)
        
        return avg_test_loss, original_images, reconstructed_images
#################################################################################################################
### ------------------------------------- VAE TRAINER --------------------------------------------------------###
#################################################################################################################


class VAETrainer(ModelTrainer):
    def __init__(self, model, optimizer, vae_loss_fn, train_loader, val_loader, device, save_path="best_vae.pth"):
        super().__init__(model, optimizer, train_loader, val_loader, device, save_path)
        self.vae_loss_fn = vae_loss_fn # Using your custom function instead of criterion

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train VAE]", unit="batch")
        
        for batch_X, _ in progress_bar: # We ignore batch_y for the VAE!
            batch_X = batch_X.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Unpack the 3 items returned by the VAE
            reconstructed_X, mu, log_var = self.model(batch_X)
            
            # Calculate loss against the ORIGINAL input (batch_X)
            loss = self.vae_loss_fn(reconstructed_X, batch_X, mu, log_var)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        return {'loss': running_loss / len(self.train_loader)}

    def validate_epoch(self, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch_X, _ in self.val_loader:
                batch_X = batch_X.to(self.device)
                
                reconstructed_X, mu, log_var = self.model(batch_X)
                loss = self.vae_loss_fn(reconstructed_X, batch_X, mu, log_var)
                
                running_loss += loss.item()
                
        return {'loss': running_loss / len(self.val_loader)}
    
    def evaluate(self, test_loader):
        self.model.eval()
        print(f"Evaluating [VAE] on the Test Set...\n")

        test_loss = 0.0
        # Instead of predictions, we will save images to visualize the VAE's performance
        original_images = []
        reconstructed_images = []

        with torch.no_grad():
            for batch_X, _ in test_loader: # Ignore labels for VAE
                batch_X = batch_X.to(self.device)
                
                # Unpack the VAE output
                reconstructed_X, mu, log_var = self.model(batch_X)
                
                # Calculate custom VAE loss
                loss = self.vae_loss_fn(reconstructed_X, batch_X, mu, log_var)
                test_loss += loss.item()
                
                # Save just the very first batch of images for plotting later
                if len(original_images) == 0:
                    original_images.extend(batch_X.cpu().numpy())
                    reconstructed_images.extend(reconstructed_X.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)

        print("="*45)
        print(" FINAL TEST METRICS: VAE ")
        print("="*45)
        print(f" Test Loss (Reconstruction + KL): {avg_test_loss:.4f}")
        print("="*45)
        
        return avg_test_loss, original_images, reconstructed_images