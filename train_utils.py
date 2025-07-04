import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_logits(output):
    """Normalize model output across architectures."""
    if isinstance(output, (tuple, list)):
        return output[0]  # e.g., GoogLeNet
    elif hasattr(output, 'logits'):
        return output.logits  # e.g., HuggingFace ViT
    return output  # Already a raw tensor (e.g., CNN)

def train_epoch(model, train_loader, criterion, optimizer, device):

    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch['image'].to(device), batch['label'].to(device)
                
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = extract_logits(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def plot_metrics(train_losses, val_losses, val_accuracies):
    """Plot training/validation metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    print(f"Training with {model.__class__.__name__}")

    """Main training loop"""
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_path = f"{model.__class__.__name__}_my_try.pth"

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device) 
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate if no improvement is registered for 5 training epochs
        if scheduler:
            scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

        # to save the model every 10 epochs
        if epoch % 10 == 0:
            epoch_model_path = f"{model.__class__.__name__}_my_try_epoch{epoch}.pth"
            torch.save(model.state_dict(), epoch_model_path)
    
    # Plot results
    plot_metrics(train_losses, val_losses, val_accuracies)
    
    return train_losses, val_losses, val_accuracies

# # Usage example:
# train_losses, val_losses, val_accuracies = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,  # Can be None if not using LR scheduling
#     num_epochs=120,
#     device=device
# )

