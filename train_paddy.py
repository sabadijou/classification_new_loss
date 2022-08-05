import torch
from utils.transformers import cifar10_transformer
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils.losses import choose_loss
import matplotlib.pyplot as plt
from configs import paddy_config as cfg
from utils.gpu import device
from backbones import RegNet
from utils.visualize_results import Visualize
import os
from dataloader.paddy_dataloader import PaddyDataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_function=None,
                 config=cfg.regnet,
                 ex_num=0):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.ex_num = ex_num
        self.dataset = PaddyDataLoader(data_path=r'D:\Datasets\image\paddy')
        self.train_loader = self.set_dataloader(self.dataset, train=True)

        self.test_loader = self.set_dataloader(self.dataset, train=False)

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
        validation_split = .2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        cuda_kwargs = {'num_workers': self.config['num_workers'],
                       'pin_memory': self.config['pin_memory'],
                       'shuffle': self.config['shuffle']}
        if train:
            train_kwargs = {'batch_size': self.config['train_batch_size'],
                            'sampler': train_sampler}
            train_kwargs.update(cuda_kwargs)
            train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
            return train_loader
        else:
            test_kwargs = {'batch_size': self.config['test_batch_size'],
                           'sampler': valid_sampler}
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
        ex_path = os.path.join(cfg.paddy['experiments_path'], str(self.ex_num), self.config['name'])
        os.makedirs(ex_path)
        os.makedirs(os.path.join(ex_path, 'ckpt'))
        os.makedirs(os.path.join(ex_path, 'results'))
        return os.path.join(ex_path, 'ckpt'), os.path.join(ex_path, 'results')


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _loss = choose_loss('ourloss', gamma=2)
    # Plot #####################################
    if not os.path.exists(cfg.paddy['experiments_path']):
        os.makedirs(cfg.paddy['experiments_path'])
    ex_num = len(os.listdir(cfg.paddy['experiments_path']))
    model = RegNet.Regnet(class_num=cfg.paddy['class_num'], pretrained=True)
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.regnet['lr'])
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.regnet['scheduler_gamma'])
    train = Trainer(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=_loss,
                    config=cfg.regnet,
                    ex_num=ex_num)
    loss_train_epoch, epoch_auc_history = train.run()