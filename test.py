import argparse
import os
from datetime import datetime

import torch
import torch.utils.data as Data
from sklearn.metrics import auc
from torchvision import transforms
from tqdm import tqdm

# Import your custom modules
from mymodel.PGNet.net import Net
from utils.data_ch3 import TestSetLoader
from utils.metrics import PD_FA, ROCMetric, mIoU, SamplewiseSigmoidMetric
from utils.util import save_test_log


def parse_args():
    """Parses command-line arguments for the testing script."""
    parser = argparse.ArgumentParser(description='Inference Script for Infrared Small Target Segmentation')
    
    # --- Path and Dataset Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint file (.pth.tar).')
    parser.add_argument('--dataset_path', type=str, default="./datasets",
                        help='Root directory of the dataset.')
    parser.add_argument('--dataset', type=str, default="SIRST",
                        help='Name of the dataset to evaluate (e.g., SIRST, NUDT-SIRST).')
    
    # --- Data Loading and Preprocessing ---
    parser.add_argument('--mode', type=str, default="txt", choices=['txt', 'folder'],
                        help='Dataset loading mode. "txt" uses a split file, "folder" uses directory structure.')
    parser.add_argument('--split', type=str, default="default",
                        help='Data split configuration (required for "txt" mode).')
    parser.add_argument('--test_txt', type=str, default="test.txt",
                        help='Name of the test split file (e.g., "test.txt").')
    parser.add_argument('--base_size', type=int, default=256,
                        help='Base size for resizing input images.')
    parser.add_argument('--resize_gt', type=str, default="no", choices=['yes', 'no'],
                        help='Set to "yes" to resize the ground truth mask for evaluation.')
    
    # --- Inference and Saving Arguments ---
    parser.add_argument('--save_pred', type=str, default='yes', choices=['yes', 'no'],
                        help='Set to "yes" to save the predicted segmentation masks.')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Batch size for testing.')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels for the model.')
    parser.add_argument('--workers', type=int, default=2, metavar='N',
                        help='Number of data loading threads.')

    return parser.parse_args()


def run_test(args):
    """
    Main function to run the model evaluation on the test dataset.
    """
    # --- 1. Setup DataLoader ---
    test_dataset = TestSetLoader(args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.workers)
    
    print("--------------- Dataset Loaded ---------------")
    print(f"Number of test samples: {len(test_dataset)}")

    # --- 2. Initialize Model and Load Weights ---
    model = Net(in_channels=args.in_channels)
    
    # Load the state dictionary safely
    checkpoint = torch.load(args.checkpoint, map_location='cpu') # Use map_location for flexibility
    
    # This handles cases where the checkpoint was saved with DataParallel
    # and has a 'module.' prefix in the keys.
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Filter out unnecessary keys and load
    model_dict = model.state_dict()
    filtered_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model_dict}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    
    model = model.cuda()
    model.eval()
    print("--------------- Model Loaded ---------------")

    # --- 3. Initialize Metrics ---
    roc_metric = ROCMetric(1, 10)
    miou_metric = mIoU()
    niou_metric = SamplewiseSigmoidMetric(1, 0)
    pd_fa_metric = PD_FA()

    # --- 4. Run Inference and Evaluation Loop ---
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, (img, mask, size, img_name) in enumerate(tbar):
            img = img.cuda()
            mask = mask.cuda()
            
            # Forward pass
            predictions = model(img)
            
            # The model might return multiple outputs (deep supervision).
            # The original code used preds[-2], this is preserved.
            # Adjust if your model's primary output is the last one (preds[-1]).
            if isinstance(predictions, (list, tuple)):
                pred = predictions[-2] 
            else:
                pred = predictions
            
            # Crop prediction to original image size
            pred = pred[:, :, :size[0], :size[1]]
                
            # Update metrics
            roc_metric.update(pred, mask)
            miou_metric.update(pred > 0.5, mask)
            niou_metric.update(pred, mask)
            pd_fa_metric.update((pred > 0.5).squeeze().cpu(), mask.squeeze().cpu(), size)

            # Update progress bar
            _, current_iou = miou_metric.get()
            current_niou = niou_metric.get()
            tbar.set_description(f'mIoU: {current_iou:.4f}, nIoU: {current_niou:.4f}')

            # Save prediction masks if enabled
            if args.save_pred == 'yes':
                pred_img = transforms.ToPILImage()((pred.squeeze(0).squeeze(0)).cpu())
                save_dir = os.path.join(os.path.dirname(args.checkpoint), args.dataset, 'predictions')
                os.makedirs(save_dir, exist_ok=True)
                pred_img.save(os.path.join(save_dir, img_name[0] + '.png'))
        
    # --- 5. Calculate Final Metrics and Save Log ---
    _, iou = miou_metric.get()
    niou = niou_metric.get()
    pd, fa = pd_fa_metric.get()
    tpr, fpr, recall, precision, _, f1_score = roc_metric.get()  
                                          
    auc_value = auc(fpr, tpr)
    
    print("\n--------------- Evaluation Finished ---------------")
    print(f"IoU: {iou:.4f}, nIoU: {niou:.4f}, F1-score: {f1_score:.4f}, AUC: {auc_value:.4f}")
    print(f"PD: {pd:.4f}, FA: {fa:.8f}")
    
    # Save detailed results to a log file
    save_test_log(os.path.dirname(args.checkpoint), iou, niou, f1_score, pd, fa, tpr, fpr, recall, precision, auc_value)
    print("--------------- Results Saved ---------------")


if __name__ == '__main__':
    args = parse_args()
    run_test(args)