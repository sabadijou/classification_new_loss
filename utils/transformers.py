import numpy
import torch
from torchvision import transforms


def cifar10_transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    return transform


def imagenet_transformer_train():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(10),
        transforms.ColorJitter(contrast=1.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform

def imagenet_transformer_val():
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform
