
cifar100 = {
            'class_num': 100,
            'experiments_path': r'experiments/Cifar100'
          }

resnet18 = {
            'name': 'resnet18',
            'dataset_name': 'Cifar100',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 200,
            'lr': 0.1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'class_num': 10,
            'weight_decay': 5e-4
            }

resnet50 = {
            'name': 'resnet50',
            'dataset_name': 'Cifar100',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 50,
            'lr': 0.001,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'weight_decay': 0.0005
            }

resnet101 = {
            'name': 'resnet18',
            'dataset_name': 'Cifar100',
            'train_batch_size': 64,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 200,
            'lr': 0.1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'class_num': 10,
            'weight_decay': 5e-4
            }


efficientnetv2 = {
            'name': 'efficientnetv2',
            'dataset_name': 'Cifar100',
            'train_batch_size': 128,
            'test_batch_size': 128,
            'num_workers': 16,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 50,
            'lr': 0.1,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'weight_decay': 0.0005
            }

mobilenetv2 = {
            'name': 'mobilenetv2',
            'dataset_name': 'Cifar100',
            'train_batch_size': 128,
            'test_batch_size': 64,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 500,
            'lr': 0.0001,
            'seed': 0,
            'seed_type': int,
            'log_interval': 0.98,
            'scheduler_gamma': 0.7,
            'weight_decay': 0.0005
            }


vgg19 = {
            'name': 'vgg19',
            'dataset_name': 'Cifar100',
            'train_batch_size': 128,
            'test_batch_size': 128,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': True,
            'dry_run': False,
            'gamma': 2,
            'epoch': 25,
            'lr': 0.001,
            'seed': 0,
            'seed_type': int,
            'log_interval': 1,
            'scheduler_gamma': 0.7,
            'weight_decay': 0.0005
            }
