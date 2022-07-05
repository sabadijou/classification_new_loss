import torch
from torchvision import datasets
from utils.transformers import cifar10_transformer
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils.losses import choose_loss
import matplotlib.pyplot as plt
from configs import cifar100 as cfg
from utils.gpu import device
from backbones import ResNet18, ResNet50, ResNet101, MobileNetV2, EfficientNetV2, VGG19
from utils.visualize_results import Visualize
import os


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_function=None,
                 config=cfg.resnet18,
                 ex_num=0):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.ex_num = ex_num
        self.train_loader = self.set_dataloader(datasets.CIFAR100('../data', train=True,
                                                                 download=True,
                                                                 transform=cifar10_transformer()),
                                                                 train=True)

        self.test_loader = self.set_dataloader(datasets.CIFAR100('../data',
                                                                train=False,
                                                                transform=cifar10_transformer()),
                                                                train=False)
        self.loss_function = loss_function
        self.epoch = 0
        self.best_auc = 0
        self.epoch_auc_history = []
        self.loss_train_epoch = []
        self.loss_test_epoch = []
        self.loss_train_batch = []
        self.ckpt_path, self.results_path = self.create_dir()

    def train_step(self):
        self.model.train()
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_function(output, target)
            self.loss_train_batch.append(loss.item())
            epoch_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

                if self.config['dry_run']:
                    break
        self.loss_train_epoch.append(sum(epoch_loss)/len(epoch_loss))

    def test(self):
        self.model.eval()
        test_loss_list = []
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss_list.append(self.loss_function(output, target).item())  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = sum(test_loss_list)/len(test_loss_list)
        self.loss_test_epoch.append(test_loss)
        self.epoch_auc_history.append(correct / 100)
        if correct > self.best_auc:
            self.best_auc = correct
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, 'best.pth'))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        print('Best Accuracy :', self.best_auc)

    def set_dataloader(self, dataset, train=True):
        cuda_kwargs = {'num_workers': self.config['num_workers'],
                       'pin_memory': self.config['pin_memory'],
                       'shuffle': self.config['shuffle']}
        if train:
            train_kwargs = {'batch_size': self.config['train_batch_size']}
            train_kwargs.update(cuda_kwargs)
            train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
            return train_loader
        else:
            test_kwargs = {'batch_size': self.config['test_batch_size']}
            test_kwargs.update(cuda_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
            return test_loader

    def run(self):
        for epoch in range(1, self.config['epoch'] + 1):
            self.train_step()
            self.test()
            self.scheduler.step()
            self.epoch = epoch

        visual = Visualize(self.loss_test_epoch,
                           self.loss_train_epoch,
                           self.loss_train_batch,
                           self.epoch_auc_history,
                           config=self.config,
                           path=self.results_path,
                           epoch_num=self.config['epoch'])
        visual.visualize_model_results()
        return self.loss_train_epoch, self.epoch_auc_history

    def create_dir(self):
        ex_path = os.path.join(cfg.cifar100['experiments_path'], str(self.ex_num), self.config['name'])
        os.makedirs(ex_path)
        os.makedirs(os.path.join(ex_path, 'ckpt'))
        os.makedirs(os.path.join(ex_path, 'results'))
        return os.path.join(ex_path, 'ckpt'), os.path.join(ex_path, 'results')


if __name__ == '__main__':
    backbone_list = ['ResNet18', 'ResNet50', 'ResNet101', 'MobileNetV2', 'EfficientNetV2', 'VGG19']
    _loss = choose_loss('ourloss', gamma=2)
    # Plot #####################################
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    if not os.path.exists(cfg.cifar100['experiments_path']):
        os.makedirs(cfg.cifar100['experiments_path'])
    ex_num = len(os.listdir(cfg.cifar100['experiments_path']))

    for backbone in backbone_list:
        if backbone == 'ResNet18':
            model = ResNet18.ResNet18(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.resnet18['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.resnet18['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.resnet18,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.resnet18['name'])
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].set_title('Loss-Cifar100')
            axs[1].plot(epoch_auc_history, label=cfg.resnet18['name'])
            axs[1].set_title('Accuracy-Cifar100')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Accuracy')
            del model, train

        elif backbone == 'ResNet50':
            model = ResNet50.ResNet50(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.resnet50['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.resnet50['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.resnet50,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.resnet50['name'])
            axs[1].plot(epoch_auc_history, label=cfg.resnet50['name'])

        elif backbone == 'ResNet101':
            model = ResNet101.ResNet101(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.resnet101['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.resnet101['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.resnet101,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.resnet101['name'])
            axs[1].plot(epoch_auc_history, label=cfg.resnet101['name'])

        elif backbone == 'MobileNetV2':
            model = MobileNetV2.MobileNetV2(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.mobilenetv2['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.mobilenetv2['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.mobilenetv2,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.mobilenetv2['name'])
            axs[1].plot(epoch_auc_history, label=cfg.mobilenetv2['name'])

        elif backbone == 'EfficientNetV2':
            model = EfficientNetV2.EfficientnetV2(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.efficientnetv2['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.efficientnetv2['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.efficientnetv2,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.efficientnetv2['name'])
            axs[1].plot(epoch_auc_history, label=cfg.efficientnetv2['name'])


        elif backbone == 'VGG19':
            model = VGG19.Vgg19(class_num=cfg.cifar100['class_num'], pretrained=False)
            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.vgg19['lr'])

            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.vgg19['scheduler_gamma'])

            train = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_function=_loss,
                            config=cfg.vgg19,
                            ex_num=ex_num)
            loss_train_epoch, epoch_auc_history = train.run()
            axs[0].plot(loss_train_epoch, label=cfg.vgg19['name'])
            axs[1].plot(epoch_auc_history, label=cfg.vgg19['name'])
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")
            fig.savefig(os.path.join(cfg.cifar100['experiments_path'], str(ex_num), 'all.png'))
            del model, train