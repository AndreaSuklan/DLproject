from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flip
    transforms.RandomAffine(
        degrees=0,  # No rotation
        shear=0.2,  # Shear range of 0.2 radians (~11.5 degrees)
        scale=(0.8, 1.2)  # Random zoom between 80%-120%
    ),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def transform_batch(example, transform):
    example['image'] = transform(example['image'])
    return example



