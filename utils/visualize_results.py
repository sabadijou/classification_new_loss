import matplotlib.pyplot as plt
import sys
import os

class Visualize:
    def __init__(self, loss_test_epoch,
                 loss_train_epoch,
                 loss_train_batch,
                 epoch_auc_history,
                 config,
                 path,
                 epoch_num=25):
        super(Visualize, self).__init__()
        self.loss_test_epoch = loss_test_epoch
        self.loss_train_batch = loss_train_batch
        self.epoch_auc_history = epoch_auc_history
        self.loss_train_epoch = loss_train_epoch
        self.epoch_num = epoch_num
        self.config = config
        self.path = path

    def visualize_model_results(self):
        # Plot AUC-Epoch
        fig, plt_2 = plt.subplots(1, 1, figsize=(5, 5))
        plt_2.plot(self.epoch_auc_history, color='red')
        plt_2.set_ylabel('Accuracy')
        plt_2.set_xlabel('Epoch')
        plt_2.set_title('Dataset: {} - Backbone: {}'.format(self.config['dataset_name'],
                                                      self.config['name']))

        fig.savefig(os.path.join(self.path, 'AUC_Epoch.png'))
        # fig.cla()

        # Plot Loss Train --> Epoch
        fig, plt_2 = plt.subplots(1, 1, figsize=(5, 5))
        plt_2.plot(self.loss_train_epoch, color='red', label='Train')
        plt_2.plot(self.loss_test_epoch, color='blue', label='Test')
        plt_2.legend(loc="upper right")
        plt_2.set_ylabel('Loss')
        plt_2.set_xlabel('Epoch')
        plt_2.set_title('Dataset: {} - Backbone: {}'.format(self.config['dataset_name'],
                                                      self.config['name']))

        fig.savefig(os.path.join(self.path, 'Loss_Epoch.png'))
        # fig.cla()

        # Plot Loss Train --> Batch
        fig, plt_2 = plt.subplots(1, 1, figsize=(5, 5))
        plt_2.plot(self.loss_train_batch, color='red')
        plt_2.set_ylabel('Loss')
        plt_2.set_xlabel('Iteration')
        plt_2.set_title('Dataset: {} - Backbone: {}'.format(self.config['dataset_name'],
                                                      self.config['name']))

        fig.savefig(os.path.join(self.path, 'Loss_Batch.png'))
        # fig.cla()
