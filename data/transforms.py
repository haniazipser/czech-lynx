from torchvision import transforms

def get_transforms(mode: str, image_size=224):
    mean = [0.485, 0.456, 0.406] #CHECK MEGADESCIPTORS DATA!!
    std  = [0.229, 0.224, 0.225]

    if mode == "baseline":
        train = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif mode == "metric_learning":
        train = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), #maybe skip?
        ])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train, val