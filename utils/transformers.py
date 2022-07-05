from torchvision import transforms


def cifar10_transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    return transform


def cifar100_transformer():
    pass


def imagenet_transformer():
    pass