import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming these are your custom utility files
from utils.metrics import ROCMetric, mIoU, SamplewiseSigmoidMetric, PD_FA
from utils.data_ch3 import TrainSetLoader, TestSetLoader
from utils.util import get_optimizer, make_dir, save_train_parser, copy_files, \
                       save_best_log, save_best_niou_log, save_train_log
from utils.loss import SoftIoULoss

# Import your model
from mymodel.PGNet.net import Net

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# For reproducibility, you can uncomment the line below
# torch.manual_seed(42)

def parse_args():
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='PyTorch Training Configuration for Infrared Small Target Segmentation')

    # --- Model and Dataset Arguments ---
    parser.add_argument('--model', type=str, default='PGNet', help='Name of the model to use.')
    parser.add_argument('--dataset', type=str, default="SIRST", help='Name of the dataset (e.g., SIRST, NUDT-SIRST).')
    parser.add_argument('--dataset_path', type=str, default="./datasets", help='Root directory of the dataset.')
    parser.add_argument('--mode', type=str, default="txt", choices=['txt', 'folder'],
                        help='Dataset loading mode. "txt" uses a split file, "folder" uses directory structure.')
    parser.add_argument('--split', type=str, default="default",
                        help='Data split configuration (required for "txt" mode).')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels for the model.')

    # --- Image Preprocessing Arguments ---
    parser.add_argument('--base_size', type=int, default=256, help='Base size to which images are resized.')
    parser.add_argument('--crop_size', type=int, default=256, help='Size of the random crop for training.')
    parser.add_argument('--resize_gt', type=str, default="no", choices=['yes', 'no'],
                        help='Whether to resize the ground truth mask during validation.')

    # --- Training and Optimization Arguments ---
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='Number of training epochs.')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='N', help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='Validation batch size.')
    parser.add_argument('--optimizer', type=str, default='Adagrad', choices=['Adam', 'Adagrad', 'AdamW', 'SGD'],
                        help='Optimizer for training.')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'],
                        help='Learning rate scheduler.')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate for schedulers like CosineAnnealingLR.')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='Number of data loading threads.')

    # --- Experiment and Logging Arguments ---
    parser.add_argument('--use_wandb', type=str, default="no", choices=['yes', 'no'],
                        help='Set to "yes" to use Weights & Biases for logging.')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results and checkpoints.')

    return parser.parse_args()


class Trainer:
    """
    The main class for handling the training and validation pipeline.
    """
    def __init__(self, args):
        self.args = args

        # --- Initialize Metrics ---
        self.roc_metric = ROCMetric(1, 10)
        self.miou_metric = mIoU()
        self.niou_metric = SamplewiseSigmoidMetric(1, 0)
        self.pd_fa_metric = PD_FA()

        # --- Setup Datasets and Dataloaders ---
        train_dataset = TrainSetLoader(args)
        test_dataset = TestSetLoader(args)
        print("--------------- Finished loading dataset ---------------")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(test_dataset)}")

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                                       shuffle=True, num_workers=args.workers, drop_last=True)
        self.val_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                                     num_workers=args.workers, drop_last=False)
        print("--------------- Finished loading dataloaders ---------------")

        # --- Setup Model, Optimizer, and Loss ---
        self.model = Net().cuda()
        self.model = nn.DataParallel(self.model)
        self.optimizer, self.scheduler = get_optimizer(self.model, self.args)

        # Define loss functions
        self.bce_loss = nn.BCELoss()
        self.soft_iou_loss = SoftIoULoss()
        
        # --- Tracking Variables ---
        self.best_iou = 0
        self.best_iou_epoch = 0
        self.best_niou = 0
        self.best_niou_epoch = 0
        self.train_loss = 0.0

    def train_one_epoch(self, epoch):
        """
        Handles the training loop for a single epoch.
        """
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = []

        if self.scheduler is not None:
            self.scheduler.step(epoch)

        for i, (img, mask, edge) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            # Forward pass
            preds = self.model(img)
            
            # Calculate loss
            # The model may return multiple predictions for deep supervision
            if isinstance(preds, (list, tuple)):
                loss = sum(self.soft_iou_loss(pred, mask) for pred in preds)
                loss += self.bce_loss(preds[-1], mask)  # Main prediction loss
            else:
                loss = self.soft_iou_loss(preds, mask) + self.bce_loss(preds, mask)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            current_lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(
                f'Epoch [{epoch:04d}/{self.args.epochs:04d}], '
                f'Train Loss: {np.mean(losses):.4f}, '
                f'LR: {current_lr:.8f}'
            )
        
        self.train_loss = np.mean(losses)

    def validate(self, epoch):
        """
        Handles the validation loop to evaluate the model's performance.
        """
        self.model.eval()
        self.miou_metric.reset()
        self.niou_metric.reset()
        self.roc_metric.reset()
        self.pd_fa_metric.reset()
        
        tbar = tqdm(self.val_loader)
        val_losses = []

        with torch.no_grad():
            for i, (img, mask, size, img_name) in enumerate(tbar):
                img, mask = img.cuda(), mask.cuda()

                # Forward pass
                preds = self.model(img)
                
                # Handle single or multiple predictions
                main_pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
                
                # Crop prediction to original image size
                # Note: Assumes size is a tuple (height, width)
                main_pred = main_pred[:, :, :size[0], :size[1]]
                
                # Calculate validation loss on the main prediction
                loss = self.soft_iou_loss(main_pred, mask)
                val_losses.append(loss.item())

                # Update metrics
                pred_binary = (main_pred > 0.5).float()
                self.miou_metric.update(pred_binary, mask)
                self.niou_metric.update(main_pred, mask)
                self.roc_metric.update(main_pred, mask)
                self.pd_fa_metric.update(pred_binary.squeeze(0).squeeze(0).cpu(), mask.squeeze(0).squeeze(0).cpu(), size)

                _, current_iou = self.miou_metric.get()
                current_niou = self.niou_metric.get()
                
                tbar.set_description(
                    f'Validation - Epoch [{epoch:04d}/{self.args.epochs:04d}], '
                    f'Val Loss: {np.mean(val_losses):.4f}, '
                    f'IoU: {current_iou * 100:.2f}%, nIoU: {current_niou * 100:.2f}%'
                )

        # --- Get final metrics for the epoch ---
        _, iou = self.miou_metric.get()
        niou = self.niou_metric.get()
        pd, fa = self.pd_fa_metric.get()
        tpr, fpr, recall, precision, _, f1_score = self.roc_metric.get()

            
        # --- Save best models and logs ---
        model_state = self.model.module.state_dict()
        val_loss_mean = np.mean(val_losses)

        if iou > self.best_iou:
            self.best_iou = iou
            self.best_iou_epoch = epoch
            save_best_log(self.args.save_path, epoch, self.train_loss, val_loss_mean, iou, niou,
                          f1_score, pd, fa, tpr, fpr, recall, precision, model_state)
            
        if niou > self.best_niou:
            self.best_niou = niou
            self.best_niou_epoch = epoch
            save_best_niou_log(self.args.save_path, epoch, self.train_loss, val_loss_mean, iou, niou,
                               f1_score, pd, fa, tpr, fpr, recall, precision, model_state)
                               
        save_train_log(self.args.save_path, epoch, self.train_loss, val_loss_mean, iou, niou,
                       self.best_iou, self.best_iou_epoch, model_state)


if __name__ == '__main__':
    args = parse_args()

    # Create directory for saving results
    args.save_path = make_dir(args.result_dir, args.split, args.dataset, args.model)
    
    # Save configuration and copy relevant files for reproducibility
    save_train_parser(args)
    copy_files(args) # Assuming a function to copy model/util files

    # Initialize W&B if enabled
    if args.use_wandb == "yes":
        wandb.init(project=f"{args.model}-{args.dataset}", config=args)

    trainer = Trainer(args=args)

    print("============= Training Started =============")
    for epoch in range(1, args.epochs + 1):
        trainer.train_one_epoch(epoch)
        trainer.validate(epoch)
    print("============= Training Finished =============")