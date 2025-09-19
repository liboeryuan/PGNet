import argparse
import torch
import torch.utils.data as Data
from tqdm import tqdm
from mymodel.PGNet.net import Net
from utils.datasets import TestSetLoader
from utils.metrics import ROCMetric05, mIoU, SamplewiseSigmoidMetric


def parse_args():
    """Parses command-line arguments for the testing script."""
    parser = argparse.ArgumentParser(
        description='Inference Script for Infrared Small Target Segmentation'
    )

    # --- Path and Dataset Arguments ---
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/lby/lby-project/PGNet/checkpoint/SIRST/best_IoU.pth.tar',
        help='Path to the trained model checkpoint file (.pth.tar).'
    )
    parser.add_argument(
        '--datasetpath',
        type=str,
        default="./datasets",
        help='Root directory of the dataset.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="SIRST",
        help='Name of the dataset to evaluate (e.g., SIRST, NUDT-SIRST).'
    )

    # --- Data Loading and Preprocessing ---
    parser.add_argument(
        '--split',
        type=str,
        default="default",
        help='Data split configuration (required for "txt" mode).'
    )
    parser.add_argument(
        '--test_txt',
        type=str,
        default="test.txt",
        help='Name of the test split file (e.g., "test.txt").'
    )
    parser.add_argument(
        '--base_size',
        type=int,
        default=256,
        help='Base size for resizing input images.'
    )
    
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=1,
        help='Batch size for testing.'
    )
    parser.add_argument(
        '--in_channels',
        type=int,
        default=1,
        help='Number of input channels for the model.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        metavar='N',
        help='Number of data loading threads.'
    )

    return parser.parse_args()


def run_test(args):
    """
    Main function to run the model evaluation on the test dataset.
    """
    # 1. Setup DataLoader
    # =========================================================================
    test_dataset = TestSetLoader(args)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False  # Typically False for testing
    )

    print("--------------- Dataset Loaded ---------------")
    print(f"Number of test samples: {len(test_dataset)}")
    print("----------------------------------------------\n")

    # 2. Initialize Model and Load Weights
    # =========================================================================
    model = Net()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Handle checkpoints saved with or without a 'state_dict' key
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'module.' prefix from keys if saved with DataParallel
    # and filter for keys present in the current model.
    model_dict = model.state_dict()
    filtered_state_dict = {
        k.replace('module.', ''): v
        for k, v in state_dict.items()
        if k.replace('module.', '') in model_dict
    }
    
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("--------------- Model Loaded ---------------")
    print(f"Model moved to device: {device}")
    print("--------------------------------------------\n")

    # 3. Initialize Metrics
    # =========================================================================
    eval_ROC = ROCMetric05(1, 10)
    eval_IoU = mIoU()
    eval_nIoU = SamplewiseSigmoidMetric(1, 0)

    # 4. Run Inference and Evaluation Loop
    # =========================================================================
    tbar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for i, (img, mask, size, img_name) in enumerate(tbar):
            img = img.to(device)
            mask = mask.to(device)

            predictions = model(img)
            pred = predictions[-2]  # Select the desired output from the model

            # Crop prediction to the original image size before padding
            h, w = size[0].item(), size[1].item()
            pred = pred[:, :, :h, :w]

            # Update metrics
            eval_ROC.update(pred, mask)
            eval_IoU.update(pred > 0.5, mask)
            eval_nIoU.update(pred, mask)

            # Update progress bar description
            _, current_iou = eval_IoU.get()
            current_niou = eval_nIoU.get()
            tbar.set_description(f'mIoU: {current_iou:.4f}, nIoU: {current_niou:.4f}')

    # 5. Calculate Final Metrics and Display Results
    # =========================================================================
    _, IoU = eval_IoU.get()
    nIoU = eval_nIoU.get()
    TPR, FPR, recall, precision, FP, F1_score = eval_ROC.get()

    print("\n------------- Evaluation Finished --------------")
    print(f"  mIoU      : {IoU:.4f}")
    print(f"  nIoU      : {nIoU:.4f}")
    print(f"  F1-score  : {F1_score:.4f}")
    print("----------------------------------------------")


if __name__ == '__main__':
    args = parse_args()
    run_test(args)