import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import json
import torch


from config.configm import get_default_config
from model.trainer_match import Trainer

# from config.config import get_default_config
# from model.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Tooth Segmentation Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test'],
                        help='Running mode: train, test, train_test')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (JSON format)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for testing or visualization')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    
    parser.add_argument('--finetune_sam', action='store_true', help='Whether to finetune SAM model')
    parser.add_argument('--training_mode', type=str, default='e2e')  
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--e2e_batch_size', type=int, default=None, help='Batch size for end-to-end mode')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--prompt_lr', type=float, default=None, help='Prompt generator learning rate')
    parser.add_argument('--sam_lr', type=float, default=None, help='SAM model learning rate')
    
    args = parser.parse_args()
    config = get_default_config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
    
    if args.checkpoint:
        config['test_checkpoint'] = args.checkpoint
    if args.resume:
        config['resume_checkpoint'] = args.resume
    if args.finetune_sam:
        config['finetune_sam'] = True
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.e2e_batch_size:
        config['e2e_batch_size'] = args.e2e_batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.prompt_lr:
        config['prompt_lr'] = args.prompt_lr
    if args.sam_lr:
        config['sam_lr'] = args.sam_lr
    
    
    print(f"Running mode: {args.mode}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Finetune SAM: {config.get('finetune_sam', False)}")
    
    trainer = Trainer(config)
    
    # Run based on mode
    if args.mode in ['train', 'train_test']:
        print("Starting training...")
        trainer.train()
        print("Training complete!")
        
    if args.mode in ['test', 'train_test']:
        print("Starting testing...")
        # If testing after training and no checkpoint is specified, use the best checkpoint
        checkpoint_to_test = config['test_checkpoint']
        if args.mode == 'train_test' and not checkpoint_to_test:
            best_checkpoint = trainer.checkpoint_dir / 'best_model.pth'
            if best_checkpoint.exists():
                checkpoint_to_test = str(best_checkpoint)
                print(f"Using best model for testing: {checkpoint_to_test}")
        
        test_loss, test_metrics, pred_dir = trainer.test(checkpoint_to_test)
        print(f"Testing complete! Average loss: {test_loss:.4f}, IoU: {test_metrics['mean_iou']:.4f}")
        print(f"Predictions saved to {pred_dir}")


if __name__ == '__main__':
    main()
