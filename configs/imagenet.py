
imagenet = {
            'class_num': 1000,
            'experiments_path': r'experiments/imagenet',
            'train_data_path': r'',
            'val_data_path': r''
          }

resnet18 = {
            'name': 'resnet18',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'class_num': 1000
            }

resnet50 = {
            'name': 'resnet50',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }

resnet101 = {
            'name': 'resnet101',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }


efficientnetv2 = {
            'name': 'efficientnetv2',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }

mobilenetv2 = {
            'name': 'mobilenetv2',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }


vgg19 = {
            'name': 'vgg19',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }

swintransformer = {
            'name': 'swintransformer',
            'dataset_name': 'imagenet',
            'train_batch_size': 64,
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
            'scheduler_gamma': 0.7
            }
