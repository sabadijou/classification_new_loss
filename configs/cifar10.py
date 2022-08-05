
cifar10 = {
            'class_num': 10,
            'experiments_path': r'experiments/cifar10'
          }

resnet18 = {
            'name': 'resnet18',
            'dataset_name': 'Cifar10',
            'train_batch_size': 128,
            'test_batch_size': 64,
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 3,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'class_num': 10
            }

resnet50 = {
            'name': 'resnet50',
            'dataset_name': 'Cifar10',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 3,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7
            }

resnet101 = {
            'name': 'resnet101',
            'dataset_name': 'Cifar10',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 3,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7
            }


efficientnetv2 = {
            'name': 'efficientnetv2',
            'dataset_name': 'Cifar10',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 3,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7
            }

mobilenetv2 = {
            'name': 'mobilenetv2',
            'dataset_name': 'Cifar10',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 25,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7
            }


vgg19 = {
            'name': 'vgg19',
            'dataset_name': 'Cifar10',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 25,
            'lr': 1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7
            }
