import torchvision.transforms as transforms


def build_train_transform(img_h, img_w):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_test_transform(img_h, img_w):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
        ]
    )
