# main.py (or notebook cell 1)
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from config import INIT_LR, NUM_EPOCHS, BATCH_SIZE, DECAY, MOMENTUM, IMG_SIZE, PATCH_SIZE

# Import from modular files
from models.SimpleCNN import CNN
from transforms import train_transform, test_transform, transform_batch
from train_utils import train_model 
from utils import setup_device, count_parameters
# import and use resnet50 and googlenet (already implemented in pytorch)
from torchvision.models import resnet50
from torchvision.models import googlenet
from torchvision.models import VisionTransformer

from transformers import ViTModel, ViTConfig, ViTForImageClassification
def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "cnn":
        return CNN(num_classes)
    elif name == "resnet":
        return resnet50()
    elif name == "googlenet":
        return googlenet()
    # !!! TODO/WORK in progress (also perhaps not necessary). Need to modify embedding size as well to use it.
    # elif name == "vit": 
    #     # Load and modify config
    #     config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    #     config.num_labels = num_classes
    #     config.image_size = IMG_SIZE  # e.g., 64
    #     config.hidden_dropout_prob = 0.1  # Optional: add any other customizations

    #     # Initialize model with the modified config and pretrained weights
    #     vit_model = ViTForImageClassification.from_pretrained(
    #         'google/vit-base-patch16-224-in21k',
    #         config=config
    #     )
    #     return vit_model
    else:
        raise ValueError(f"Unsupported model name: {name}")

def prepare_dataset():
    ds = load_dataset("blanchon/EuroSAT_RGB")
    ds["train"] = ds["train"].map(lambda x: transform_batch(x, train_transform))
    ds["test"] = ds["test"].map(lambda x: transform_batch(x, test_transform))
    ds["train"].set_format(type='torch', columns=['image', 'label'])
    ds["test"].set_format(type='torch', columns=['image', 'label'])

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE)
    num_classes = len(ds["train"].features["label"].names)
    return train_loader, test_loader, num_classes

def main(model_names):
    device = setup_device()
    train_loader, test_loader, num_classes = prepare_dataset()
    for model_name in model_names:
        print(f"\n--- Training model: {model_name.upper()} ---")
        model = get_model(model_name, num_classes).to(device)

        print(f"Trainable parameters: {count_parameters(model)}")

        optimizer = SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM,
                        weight_decay=DECAY, nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
            device=device
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one or more models on EuroSAT_RGB dataset")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Specify models to train. Options: cnn, resnet, googlenet, vit")
    args = parser.parse_args()
    main(args.models)



# def main():
#     # Setup device
#     device = setup_device()
    
#     # Load and prepare dataset
#     ds = load_dataset("blanchon/EuroSAT_RGB")
    
#     # Apply transforms
#     train_dataset = ds['train'].map(lambda x: transform_batch(x, train_transform))
#     test_dataset = ds['test'].map(lambda x: transform_batch(x, test_transform))
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
#     # Initialize model
#     num_classes = len(ds['train'].features['label'].names)



#     criterion = nn.CrossEntropyLoss()
    
#     # Convert dataset formats
#     train_dataset.set_format(type='torch', columns=['image', 'label'])
#     test_dataset.set_format(type='torch', columns=['image', 'label'])
    
    ## Train basic CNN 
    # model = CNN(num_classes=num_classes).to(device)
    
    # # Setup training components
    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=INIT_LR, 
    #     momentum=MOMENTUM, 
    #     weight_decay=DECAY, 
    #     nesterov=True
    # )
    # scheduler = ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',
    #     factor=0.1,
    #     patience=5
    # )

    # train_losses, val_losses, val_accuracies = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     num_epochs=NUM_EPOCHS,
    #     device=device
    # )

    # ## Train ResNet50

    # model2=resnet50().to(device) 

    # optimizer2 = torch.optim.SGD(
    #     model2.parameters(), 
    #     lr=INIT_LR, 
    #     momentum=MOMENTUM, 
    #     weight_decay=DECAY, 
    #     nesterov=True
    # )

    # scheduler2 = ReduceLROnPlateau(
    #     optimizer2, 
    #     mode='min',
    #     factor=0.1,
    #     patience=5
    # )

    # resnet50_train_losses, resnet50_val_losses, resnet50_val_accuracies = train_model(
    #     model=model2,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=optimizer2,
    #     scheduler=scheduler2,
    #     num_epochs=NUM_EPOCHS,
    #     device=device
    # )

    # ## Train GoogleNet 

    # model3=googlenet().to(device)     

    # optimizer3 = torch.optim.SGD(
    #     model3.parameters(), 
    #     lr=INIT_LR, 
    #     momentum=MOMENTUM, 
    #     weight_decay=DECAY, 
    #     nesterov=True
    # )
    # scheduler3 = ReduceLROnPlateau(
    #     optimizer3, 
    #     mode='min',
    #     factor=0.1,
    #     patience=5
    # )

    # googlenet_train_losses, googlenet_val_losses, googlenet_val_accuracies = train_model(
    #     model=model3,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=optimizer3,
    #     scheduler=scheduler3,
    #     num_epochs=NUM_EPOCHS,
    #     device=device
    # )
    

    # Train VisionTransformer 
    # model4=VisionTransformer(image_size=IMG_SIZE, patch_size=PATCH_SIZE)
    # config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    # config.image_size = IMG_SIZE 
    # model4 = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    # model4 = ViTModel(config)

    # optimizer4 = torch.optim.SGD(
    #     model4.parameters(), 
    #     lr=INIT_LR, 
    #     momentum=MOMENTUM, 
    #     weight_decay=DECAY, 
    #     nesterov=True
    # )

    # scheduler4 = ReduceLROnPlateau(
    #     optimizer4, 
    #     mode='min',
    #     factor=0.1,
    #     patience=5
    # )

    # vit_train_losses, vit_val_losses, vit_val_accuracies = train_model(
    #     model=model4,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=optimizer4,
    #     scheduler=scheduler4,
    #     num_epochs=NUM_EPOCHS,
    #     device=device
    # )

# if __name__ == "__main__":
    # main() 
