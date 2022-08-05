
paddy = {
            'class_num': 10,
            'experiments_path': r'experiments/paddy'
          }

regnet = {
            'name': 'regnet',
            'dataset_name': 'Paddy Kaggle',
            'train_batch_size': 128,
            'test_batch_size': 128,
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': False,
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