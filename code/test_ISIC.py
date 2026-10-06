import argparse
import os
import random
import logging
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
import torchmetrics
import torch.nn.functional as F

from dataloaders.dataset_ISIC import ISIC2018_dataset
from redino.NIX_ISIC import Dino_seg


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ISIC2018', help='Dataset root path')
parser.add_argument('--exp', type=str, default='ISIC', help='Name of Experiment')
parser.add_argument('--epoch', type=str, default='model', help='Test model')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--input_size', type=int, default=256, help='input size of network')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1314, help='random seed')
args = parser.parse_args()


def get_binary_metrics(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = torchmetrics.MetricCollection(
        {
            "F1_Dice": torchmetrics.classification.BinaryF1Score(threshold=0.5),
            "Accuracy": torchmetrics.classification.BinaryAccuracy(threshold=0.5),
            "Precision": torchmetrics.classification.BinaryPrecision(threshold=0.5),
            "Specificity": torchmetrics.classification.BinarySpecificity(threshold=0.5),
            "Sensitivity_Recall": torchmetrics.classification.BinaryRecall(threshold=0.5),
            "IoU_Jaccard": torchmetrics.classification.BinaryJaccardIndex(threshold=0.5),
        }
    ).to(device)

    return metrics


def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Dino_seg(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load("../model/" + args.exp + "/{}.pth".format(args.epoch), weights_only=True))
    model.eval()

    db_test = ISIC2018_dataset(base_dir=args.root_path, split="test", img_size=args.input_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    metrics = get_binary_metrics(device)
    metrics.reset()

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
            image = sampled_batch["image"].to(device).float()
            label = sampled_batch["label"].to(device)

            if label.ndim == 4:
                label = label.squeeze(1)
            label = (label > 0).long()

            logits = model(image)

            if logits.shape[-2:] != label.shape[-2:]:
                logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
            if logits.shape[1] == 2:
                prob = torch.softmax(logits, dim=1)[:, 1]
            elif logits.shape[1] == 1:
                prob = torch.sigmoid(logits[:, 0])
            else:
                raise ValueError(
                    f"Unsupported logits shape: {logits.shape}. "
                    f"Expected channel number 1 or 2."
                )

            pred = (prob >= 0.5).long()

            pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred_np = binary_opening(pred_np,structure=np.ones((3, 3))).astype(np.uint8)
            pred_np = binary_fill_holes(pred_np).astype(np.uint8)

            pred = torch.from_numpy(pred_np).unsqueeze(0).to(device).long()
            metrics.update(pred, label)

    results = metrics.compute()

    logging.info(
        f"{args.epoch}: "
        f"F1/Dice: {results['F1_Dice']:.4f}  "
        f"Sensitivity/Recall: {results['Sensitivity_Recall']:.4f}  "
        f"Specificity: {results['Specificity']:.4f}  "
        f"Precision: {results['Precision']:.4f}  "
        f"IoU/Jaccard: {results['IoU_Jaccard']:.4f}  "
        f"Accuracy: {results['Accuracy']:.4f}  "
        f"threshold: 0.5"
    )
    return "Testing Finished!"


if __name__ == '__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../test/{}".format(args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+'/'+"log.txt", level=logging.INFO, format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    inference()


