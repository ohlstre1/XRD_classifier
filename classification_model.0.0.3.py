import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import argparse

class XRDClassifier(nn.Module):
    def __init__(self, input_length=4500, num_classes=2):
        super(XRDClassifier, self).__init__()
        
        # 1D CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output(input_length)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_classes)
    
    def _get_conv_output(self, shape):
        # Helper function to calculate the size of the flattened features
        bs = 1
        input = torch.rand(bs, 1, shape)
        output = self._forward_conv(input)
        return int(np.prod(output.size()))
    
    def _forward_conv(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        return x
    
    def forward(self, x):
        # Complete forward pass
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # No log_softmax since we're using CrossEntropyLoss
        return x

class XRDDataHandler:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def load_data(self, dataset_path, sample_limit=None):
        dataset_dict = torch.load(dataset_path, map_location=self.device)
        synth_xrd = dataset_dict["synth_xrd"]
        real_xrd = dataset_dict["real_xrd"]
        
        # We'll ignore any existing labels since we'll use custom labels
        # as specified in the main function
        labels = None
        
        if sample_limit is not None:
            synth_xrd = synth_xrd[:sample_limit]
            real_xrd = real_xrd[:sample_limit]
        
        print(f"Loaded dataset with {len(synth_xrd)} synthetic XRD patterns and {len(real_xrd)} real XRD patterns")
        
        return synth_xrd, real_xrd, labels
    
    def prepare_datasets(self, synth_xrd, real_xrd, labels, train_ratio=0.8, val_ratio=0.2):
        # Make sure labels match the number of samples
        if len(labels) > len(synth_xrd):
            labels = labels[:len(synth_xrd)]
        
        # Create datasets for synthetic and real data
        synth_dataset = self._create_dataset(synth_xrd, labels)
        real_dataset = self._create_dataset(real_xrd, labels)  # Using same labels for real data
        
        # Split synthetic dataset for training and validation
        train_size = int(train_ratio * len(synth_dataset))
        val_size = len(synth_dataset) - train_size  # Use all remaining samples for validation
        
        train_dataset, val_dataset = random_split(
            synth_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Use real data for testing
        test_dataset = real_dataset
        
        print(f"Dataset split: Train={len(train_dataset)} (synthetic), Val={len(val_dataset)} (synthetic), Test={len(test_dataset)} (real)")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_dataset(self, xrd_data, labels):
        class SimpleXRDDataset(Dataset):
            def __init__(self, xrd_data, labels):
                self.xrd_data = xrd_data.clone().detach().float().unsqueeze(1) if torch.is_tensor(xrd_data) else torch.tensor(xrd_data, dtype=torch.float32).unsqueeze(1)
                self.labels = labels.clone().detach() if torch.is_tensor(labels) else torch.tensor(labels)
            
            def __len__(self):
                return len(self.xrd_data)
            
            def __getitem__(self, idx):
                return self.xrd_data[idx], self.labels[idx]
        
        return SimpleXRDDataset(xrd_data, labels)
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        # Get values with defaults if not in config
        num_workers = wandb.config.get('num_workers', 0)
        pin_memory = wandb.config.get('pin_memory', False)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_dataloader, val_dataloader, test_dataloader

class XRDModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # Log model architecture to wandb
        wandb.watch(self.model, log="all")
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate, weight_decay):
        # Use CrossEntropyLoss instead of NLLLoss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=wandb.config.get('lr_patience', 5), 
            verbose=True
        )
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
                
                # Log batch metrics every 10 batches
                if batch_idx % 10 == 0:
                    wandb.log({
                        "batch/train_loss": loss.item(),
                        "batch/train_accuracy": 100.0 * (predicted == targets).sum().item() / targets.size(0),
                        "batch/learning_rate": optimizer.param_groups[0]['lr'],
                        "batch/global_step": epoch * len(train_loader) + batch_idx
                    })
                
                progress_bar.set_postfix({"loss": loss.item()})
            
            train_loss /= train_total
            train_accuracy = 100.0 * train_correct / train_total
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint to wandb
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, "best_model.pth")
                wandb.save("best_model.pth")
            
            # Log epoch metrics
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return train_losses, val_losses, val_accuracies
    
    def evaluate(self, data_loader, criterion=None):
        if criterion is None:
            # Use CrossEntropyLoss for evaluation too
            criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_confidences = []  # Store confidence scores
        
        test_loss = 0.0
        # Fixed: Using CrossEntropyLoss instead of NLLLoss
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
                # Get predictions and probabilities
                # Fixed: Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Extract confidence scores for the predicted classes
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    if i < len(predicted):  # Safety check
                        pred_idx = predicted[i].item()
                        if pred_idx < probs[i].size(0):  # Another safety check
                            confidence = probs[i][pred_idx].item()
                            all_confidences.append(confidence)
                
                # Extend prediction and target lists
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Log a few example predictions to wandb
                if len(all_preds) <= 50:  # Only log the first few predictions
                    for i in range(min(len(predicted), len(targets))):
                        if i < len(predicted) and i < len(targets):  # Safety check
                            pred_idx = predicted[i].item()
                            confidence = probs[i][pred_idx].item() if pred_idx < probs[i].size(0) else 0.0
                            wandb.log({
                                "predictions": wandb.Table(
                                    columns=["True", "Predicted", "Confidence"],
                                    data=[[int(targets[i].item()), 
                                          int(predicted[i].item()), 
                                          float(confidence)]]
                                )
                            })
        
        # Calculate average test loss
        test_loss /= len(test_loader.dataset)
        wandb.log({"test/avg_loss": test_loss})
        
        # Log prediction probabilities histogram if we have confidences
        if all_confidences:
            try:
                wandb.log({"prediction_confidence": wandb.Histogram(
                    np_histogram=np.histogram(
                        all_confidences,
                        bins=20,
                        range=(0, 1)
                    )
                )})
            except Exception as e:
                print(f"Warning: Could not log confidence histogram: {e}")
        
        return all_preds, all_targets
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        if 'best_val_accuracy' in checkpoint:
            self.best_val_accuracy = checkpoint['best_val_accuracy']

def plot_results(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    
    # Save locally and to wandb
    training_plot_path = 'xrd_classifier_training.png'
    plt.savefig(training_plot_path)
    wandb.log({"training_plots": wandb.Image(training_plot_path)})
    plt.close()

def plot_confusion_matrix(y_true, y_pred, num_classes=None):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Fixed: Uncommenting visualization code if num_classes is small enough to visualize
    max_classes_to_plot = 50  # Only plot confusion matrix if classes <= this number
    
    # If there are too many classes, limit to the ones present in the data
    if num_classes is None or num_classes > 20:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        class_indices = unique_classes
    else:
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        class_indices = range(num_classes)
    
    # Only create visual plot if number of classes is manageable
    if len(class_indices) <= max_classes_to_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_indices, yticklabels=class_indices)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save locally and to wandb
        confusion_matrix_path = 'xrd_confusion_matrix.png'
        plt.savefig(confusion_matrix_path)
        wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})
        plt.close()
    else:
        print(f"Skipping visual confusion matrix plot - too many classes ({len(class_indices)} > {max_classes_to_plot})")
    
    # Always log confusion matrix as a table in wandb
    try:
        wandb.log({"confusion_matrix_table": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=[str(c) for c in class_indices])
        })
    except Exception as e:
        print(f"Warning: Could not log confusion matrix to wandb: {e}")

def main():
    # Parameters are now all taken from wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset parameters
    dataset_path = wandb.config.get('dataset_path', "xrd_dataset_labeled_dtw_window.pt")
    sample_limit = wandb.config.get('sample_limit', 1000)
    batch_size = wandb.config.get('batch_size', 32)
    
    # Model parameters
    num_classes = wandb.config.get('num_classes', sample_limit)
    
    # Training parameters
    num_epochs = wandb.config.get('epochs', 50)
    learning_rate = wandb.config.get('learning_rate', 0.001)
    weight_decay = wandb.config.get('weight_decay', 1e-5)
    
    # Initialize data handler
    data_handler = XRDDataHandler(device)
    
    # Load data but ignore the labels from the dataset
    synth_xrd, real_xrd, _ = data_handler.load_data(dataset_path, sample_limit)
    
    # Create labels from 0 to num_classes-1, ensuring we don't exceed the data size
    max_labels = min(num_classes, len(synth_xrd))
    labels = torch.arange(0, max_labels, 1)
    if len(synth_xrd) > max_labels:
        # If we have more data than labels, repeat labels as needed
        repeats = len(synth_xrd) // max_labels + 1
        labels = labels.repeat(repeats)[:len(synth_xrd)]
    
    # Prepare datasets - training on synth_xrd and testing on real_xrd
    train_ratio = wandb.config.get('train_ratio', 0.8)
    val_ratio = wandb.config.get('val_ratio', 0.2)
    train_dataset, val_dataset, test_dataset = data_handler.prepare_datasets(
        synth_xrd, real_xrd, labels, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio
    )
    
    train_loader, val_loader, test_loader = data_handler.create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Log dataset information
    wandb.log({
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "num_classes": num_classes,
        "actual_num_classes": max_labels
    })
    
    # Initialize model
    input_length = synth_xrd.shape[1] if len(synth_xrd.shape) > 1 else synth_xrd.shape[0]
    model = XRDClassifier(input_length=input_length, num_classes=max_labels).to(device)
    
    # Train model
    trainer = XRDModelTrainer(model, device)
    train_losses, val_losses, val_accuracies = trainer.train(
        train_loader, val_loader, 
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Plot training results
    plot_results(train_losses, val_losses, val_accuracies)
    
    # Test on real data
    print("\nEvaluating model on real XRD data...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Log test metrics
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_accuracy
    })
    
    # Get predictions for confusion matrix
    predictions, targets = trainer.test(test_loader)
    plot_confusion_matrix(targets, predictions, max_labels)
    
    # Additional performance metrics
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    
    # Only compute detailed metrics if not too many classes
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='macro')
        
        # Log additional metrics
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1
        })
        
        # Generate classification report
        report = classification_report(targets, predictions)
        print("\nClassification Report:")
        print(report)
        
        # Log classification report if not too many classes
        if max_labels <= 100:  # Arbitrary limit to avoid overwhelming wandb
            wandb.log({"classification_report": wandb.Table(
                columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                data=[[str(c), p, r, f, s] for c, (p, r, f, s) in 
                    enumerate(zip(*precision_recall_fscore_support(targets, predictions)))]
            )})
    except Exception as e:
        print(f"Warning: Could not compute detailed metrics: {e}")
    
    # Save the model
    model_path = 'xrd_classifier_model.pth'
    trainer.save_model(model_path)
    wandb.save(model_path)
    print(f"Model saved as '{model_path}'")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    # Set up command line arguments for wandb configuration
    parser = argparse.ArgumentParser(description='XRD Classification with Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='xrd-classification',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--sample-limit', type=int, default=1000,
                        help='Limit number of samples from dataset (None to use all)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (default: same as sample-limit)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Use pinned memory for data loading (faster on GPU)')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    
    args = parser.parse_args()
    
    # Set num_classes to sample_limit if not specified
    if args.num_classes is None:
        args.num_classes = args.sample_limit
    
    # Initialize wandb with command line arguments
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={
            "architecture": "XRDClassifier-1DCNN",
            "dataset_path": "data/xrd_dataset_labeled_dtw_window.pt",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "sample_limit": args.sample_limit,
            "num_classes": args.num_classes,
            "train_ratio": 0.99,
            "val_ratio": 0.01,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "lr_patience": args.lr_patience
        }
    )
    
    main()