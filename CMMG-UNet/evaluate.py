import argparse
from engine.wrapper import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset import QaTa
import utils.config as config
from thop import profile, clever_format

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

if __name__ == '__main__':

    args = get_parser()
    dataset_name = 'mosmed'
    # load model        
    model = KMMGUNet_Wrapper(args)
    checkpoint = torch.load(f'../autodl-tmp/save_model/{args.model_save_filename}-v1.ckpt',\
                    map_location='cpu')["state_dict"]
    # checkpoint = torch.load(f'./save_model/{args.model_save_filename}-v1.ckpt',\
                    # map_location='cpu')["state_dict"]
    # model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(checkpoint, strict=True)
    
    try:
        mb = model.model.memory_bank
        
        print(f"\n{'='*30} Memory Bank Integrity Check {'='*30}")
        print(f"Memory Capacity (N) : {mb.N}")
        print(f"Current Pointer     : {mb.current_ptr.item()}")
        print(f"Is Full Status      : {mb.is_full.item()}  (Expected 1 for trained models)")
        
        text_data_sum = mb.text_bank.abs().sum().item()
        if text_data_sum == 0:
            print(f"Text Bank Data      : ALL ZEROS (Empty! Checkpoint might be from pre-warmup epoch)")
        else:
            print(f"Text Bank Data      :  Loaded (Non-zero, sum={text_data_sum:.2f})")
            
        if hasattr(mb, 'visual_bank_3'):
            vis_data_sum = mb.visual_bank_3.abs().sum().item()
            status = " Empty" if vis_data_sum == 0 else f" Loaded (sum={vis_data_sum:.2f})"
            print(f"Visual Bank 3 Data  : {status}")
            
        print(f"{'='*80}\n")
        
    except AttributeError as e:
        print(f"\n Could not inspect Memory Bank: {e}\n")
 
 
    # dataloader
    ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(accelerator='gpu',devices=1,precision=16,) 
    model.eval()
    trainer.test(model, dl_test) 
