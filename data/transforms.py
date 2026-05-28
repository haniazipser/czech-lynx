from torchvision import transforms

def get_transforms(mode: str, image_size=224):
    mean_efficient_net = [0.485, 0.456, 0.406]
    std_efficient_net  = [0.229, 0.224, 0.225]

    mean_mega_descriptor = [0.5, 0.5, 0.5] #to be confirmed
    std_mega_descriptor = [0.5, 0.5, 0.5]

    if mode == "baseline":
        train = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean_efficient_net, std_efficient_net),

        ])
        val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean_efficient_net, std_efficient_net),
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
            transforms.Normalize(mean_mega_descriptor, std_mega_descriptor)
        ])

        val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean_mega_descriptor, std_mega_descriptor),
        ])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return train, val