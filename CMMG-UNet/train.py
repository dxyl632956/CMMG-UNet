import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
import utils.config as config
from engine.wrapper import *
import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation - MMI-UNet')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='random seed for reproducibility')
    
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    

    cfg.seed = args.seed
    return cfg

def main():

    args = get_parser()

    pl.seed_everything(args.seed, workers=True)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using Seed: {args.seed}")


    ds_train = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = QaTa(csv_path=args.valid_csv_path,
                    root_path=args.valid_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        

    
    model = KMMGUNet_Wrapper(args)


    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
        save_weights_only=True 
    )


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )


    csv_logger = CSVLogger(save_dir="logs/", name=args.model_save_filename)


    trainer = pl.Trainer(
        logger=csv_logger,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu', 
        devices=args.device,
        callbacks=[model_ckpt, early_stopping],
        enable_progress_bar=True,  
        deterministic=False, 
        gradient_clip_val=0.5,  
        precision=16,
        accumulate_grad_batches=2

    ) 

    print(f"Starting training: {args.model_save_filename}...")
    trainer.fit(model, dl_train, dl_valid)
    print('Training process completed successfully.')

if __name__ == '__main__':
    main()