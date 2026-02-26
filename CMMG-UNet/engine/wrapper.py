from utils.model import *
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import cv2
import sys
import numpy as np
import datetime
import time


class KMMGUNet_Wrapper(pl.LightningModule):
    def __init__(self, args):
        super(KMMGUNet_Wrapper, self).__init__()
        self.warmup_epochs = args.warmup_epochs

        self.model = KMMG_UNet(
            bert_type=args.bert_type, 
            vision_type=args.vision_type, 
            project_dim=args.project_dim,
            memory_N=args.memory_N,  
            memory_K=args.memory_K,    
            warmup_epochs=self.warmup_epochs,
            cluster_k=args.cluster_k
        )
        self.mem_buffer_text = []
        self.mem_buffer_visual = []

        pretrained_path = './pre-trained/convnext_tiny_22k_224.pth'
        print(f"Loading pretrained backbone from {pretrained_path}...")
        
        try:
            pre_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model' in pre_dict:
                pre_dict = pre_dict['model']
            elif 'state_dict' in pre_dict:
                pre_dict = pre_dict['state_dict']
              
            model_dict = self.model.state_dict()
            matched_dict = {}
            skipped_keys = []
          
            for k, v in pre_dict.items():

                clean_k = k.replace('convnext.', '').replace('module.', '').replace('backbone.', '')
                if clean_k in model_dict:
                    if v.shape == model_dict[clean_k].shape:
                        matched_dict[clean_k] = v
                    else:
                        skipped_keys.append((k, f"Shape mismatch: pretrained {v.shape} vs model {model_dict[clean_k].shape}"))
                else:

                    for model_key in model_dict.keys():
                        if 'downsample_layers' in model_key:

                            parts = k.split('.')
                            if len(parts) > 2 and parts[0] in ['stages', 'downsample_layers']:
                                new_parts = parts.copy()
                                if parts[0] == 'stages':
                                    new_parts[0] = 'stages' if int(parts[1]) < 4 else 'downsample_layers'
                                if '.'.join(new_parts[1:]) in model_dict:
                                    matched_dict['.'.join(new_parts[1:])] = v
                                    break
                  
            model_dict.update(matched_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f'Successfully loaded {len(matched_dict)} layers into Encoder.')
            if skipped_keys:
                print(f'Skipped {len(skipped_keys)} layers due to shape mismatch or other issues.')
                for i, (key, reason) in enumerate(skipped_keys[:10]):
                    print(f'  {i+1}. {key}: {reason}')
                if len(skipped_keys) > 10:
                    print(f'  ... and {len(skipped_keys) - 10} more')
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights. Error: {e}")
        
        self.lr = args.lr
        self.history = {}
        self.loss_fn = DiceCELoss(sigmoid=True,smooth_nr=1e-5, smooth_dr=1e-5)
        metrics_dict = {"acc":Accuracy(task='binary'), "dice":Dice(), "MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,weight_decay=0.05,eps=1e-8)
        max_epochs = self.hparams.args.max_epochs if hasattr(self.hparams, 'args') else 200
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}

    # def on_train_epoch_start(self):
    #     current_epoch = self.current_epoch
    #     warmup_limit = self.warmup_epochs

    #     if current_epoch >= warmup_limit:
    #         self._freeze_encoder_strategy(current_epoch, warmup_limit)

    # def _freeze_encoder_strategy(self, current_epoch, warmup_limit):
    #     net = self.model 
        

    #     for param in net.downsample_layers.parameters():
    #         param.requires_grad = False
    #     for param in net.stages.parameters():
    #         param.requires_grad = False
            

    #     for param in net.text_encoder.parameters():
    #         param.requires_grad = False
            

    #     net.downsample_layers.eval()
    #     net.stages.eval()
    #     net.text_encoder.eval()

    #     if current_epoch == warmup_limit:
    #         print(f"\n{'='*25} STRATEGY SWITCH: FREEZING ENCODER {'='*25}")
    #         print(f"Current Epoch: {current_epoch} | Warmup Phase Finished.")
    #         print(f"Backbone & Text Encoder are now FROZEN.")
    #         print(f"{'='*80}\n")
    # # ----------------------------------------------------
    def forward(self, x):
        return self.model.forward(x, current_epoch=0)

    def shared_step(self, batch, batch_idx):
        x, y = batch

        logits, mem_collect = self.model(x, target=y, current_epoch=self.current_epoch)
        
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.loss_fn(logits.float(), y.float())

 
        if self.training and mem_collect is not None and self.current_epoch >= self.warmup_epochs:
            self.model.memory_bank.update_wld(loss.detach().item(), mem_collect[0], mem_collect[1])

        return {'loss': loss, 'preds': logits.detach(), 'y': y.detach()}
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)
  
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)

        # x, y = batch
        # image, text = x
        # pred_raw = self.model([image, text], current_epoch=100)
        # img_h = torch.flip(image, dims=[3])
        # pred_h = self.model([img_h, text], current_epoch=100)
        # pred_h = torch.flip(pred_h, dims=[3])
        # img_v = torch.flip(image, dims=[2])
        # pred_v = self.model([img_v, text], current_epoch=100)
        # pred_v = torch.flip(pred_v, dims=[2])
        # preds = (pred_raw + pred_h + pred_v) / 3.0
        # loss = self.loss_fn(preds, y)
        # return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}
  
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        else:
            return self(batch)
      
    def shared_step_end(self, outputs, stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name, step_metric, prog_bar=True)
        return outputs["loss"].mean()
      
    def training_step_end(self, outputs):
        return {'loss': self.shared_step_end(outputs, "train")}
          
    def validation_step_end(self, outputs):
        return {'val_loss': self.shared_step_end(outputs, "val")}
          
    def test_step_end(self, outputs):
        return {'test_loss': self.shared_step_end(outputs, "test")}
          
    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        epoch = self.current_epoch
        stage_loss = torch.mean(torch.tensor([t.get((stage+"_loss").replace('train_',''), 0) for t in outputs])).item()
        dic = {"epoch": epoch, stage+"_loss": stage_loss}
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)  
        return dic 
  
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")

        if self.current_epoch == self.model.warmup_epochs - 1:
            self.model.memory_bank.reset()
        self.print(dic)
        
        ptr = self.model.memory_bank.current_ptr.item()
        total = self.model.memory_bank.N
        self.print(f"\n[Memory Check] Slot Occupancy: {ptr}/{total} ({(ptr/total)*100:.1f}%)")
    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")

        opt = self.trainer.optimizers[0]
        current_lr = opt.param_groups[0]['lr']
        dic['learning_rate'] = current_lr
        

        epoch = self.current_epoch
        if epoch in self.history:
            self.history[epoch].update({'learning_rate': current_lr})
        else:
            self.history[epoch] = {'learning_rate': current_lr}
        

        df = pd.DataFrame(self.history.values())
        target_cols = ['epoch', 'learning_rate', 
                       'train_loss', 'val_loss', 
                       'train_dice', 'val_dice', 
                       'train_MIoU', 'val_MIoU']
        cols = [c for c in target_cols if c in df.columns] + [c for c in df.columns if c not in target_cols]
        if not df.empty:
            df = df[cols]
            df.to_csv('training_log.csv', index=False)
      
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)
      
        # Best Score Logging
        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb:
            monitor = ckpt_cb.monitor 
            mode = ckpt_cb.mode 
            if not df.empty and monitor in df.columns:
                arr_scores = df[monitor].values
                best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
                if best_score_idx == len(arr_scores)-1: 
                    self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor, arr_scores[best_score_idx]), file=sys.stderr)
  
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)
      
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
  
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)


def load_pretrained_weights_complete(model, pretrained_path, verbose=True):

    print(f"Loading pretrained weights from: {pretrained_path}")
    
    try:

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        

        if 'model' in checkpoint:
            pre_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pre_dict = checkpoint['state_dict']
        else:
            pre_dict = checkpoint
        

        model_dict = model.state_dict()
        

        matched_keys = []
        unmatched_keys = []
        

        for pre_key, pre_value in pre_dict.items():

            clean_key = pre_key
            

            for prefix in ['module.', 'backbone.', 'encoder.', 'convnext.', 'model.']:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            
   
            if clean_key in model_dict:
                if model_dict[clean_key].shape == pre_value.shape:
                    model_dict[clean_key] = pre_value
                    matched_keys.append((pre_key, clean_key))
                else:
                    print(f"Shape mismatch for {clean_key}: "
                          f"pretrained {pre_value.shape}, model {model_dict[clean_key].shape}")
                    unmatched_keys.append(pre_key)
            else:
                unmatched_keys.append(pre_key)
        

        model.load_state_dict(model_dict, strict=False)
        

        total_pretrained = len(pre_dict)
        print(f"Successfully loaded {len(matched_keys)}/{total_pretrained} layers")
        print(f"Model has {len(model_dict)} parameters total")
        print(f"Unmatched keys: {len(unmatched_keys)}")
        
        if verbose and len(unmatched_keys) > 0:
            print("\nUnmatched keys (first 20):")
            for key in unmatched_keys[:20]:
                print(f"  {key}")
            if len(unmatched_keys) > 20:
                print(f"  ... and {len(unmatched_keys) - 20} more")
        
        return model
    
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return model