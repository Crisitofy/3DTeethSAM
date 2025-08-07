def get_default_config():
    """获取默认配置"""
    return {

        'data_dir': 'preprocess_data',  
        'h5_path': 'preprocess_data/h5/dataset.h5', 

        'train_split_path': 'preprocess/split/official/training_all.txt',
        'val_split_path': 'preprocess/split/official/testing_all.txt',
        'test_split_path': 'preprocess/split/official/testing_all.txt',
        'save_dir': 'results',

        'batch_size': 4,
        'num_workers': 6,
        'epochs': 100,
        'weight_decay': 5e-4,
        'log_freq': 10,
        'vis_freq': 50, # 1
        'save_freq': 10,
        'val_interval': 2,
        'warmup_epochs': 5,
        
        'dropout_rate': 0.35,
        'embed_dim': 256,            
        'num_teeth': 16,            
        'num_classes': 17,            
        
        'training_mode': 'e2e',       
        
        'learning_rate': 1e-4,
        'prompt_lr':1e-4,           
        'uncertainty_lr': 1e-3,
        'sam_lr': 1e-5,               
        'grad_clip': 1.0,         
        
        'finetune_sam': False,     
        'freeze_refine_net': False,    
        'freeze_prompt_generator': False,
        
        'sam_config': 'sam2.1_hiera_l.yaml',
        'sam_checkpoint': 'model/sam2/checkpoints/sam2.1_hiera_large.pt',

        'scheduler_patience': 5,
        'scheduler_factor': 0.7,
        
        'resume_checkpoint': None,
        'test_checkpoint': None,
        
        'use_amp': True,
        'amp_dtype': 'bfloat16',

        # select view(None means all views)
        # for example, set [0, 1, 2, 3] to use the first 4 views
        # or set [3, 4, 5, 6] to use the last 3 views
        'view_indices': [0, 1, 2, 3, 4, 5, 6]
    }
