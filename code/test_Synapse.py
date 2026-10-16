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
from utils import test_single_volume

from dataloaders.dataset_Synapse import Synapse_dataset
from redino.NIX_Synapse import Dino_seg


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse', help='Dataset root path')
parser.add_argument('--exp', type=str, default='Synapse', help='Name of Experiment')
parser.add_argument('--epoch', type=str, default='model', help='Test model')
parser.add_argument('--num_classes', type=int,  default=9, help='output channel of network')
parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224], help='input size of network')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1314, help='random seed')
args = parser.parse_args()


def inference():
    model = Dino_seg(num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load("../model/" + args.exp + "/{}.pth".format(args.epoch), weights_only=True))
    model.eval()

    db_test = Synapse_dataset(base_dir=args.root_path, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=args.input_size)
        metric_list += np.array(metric_i)
        logging.info('idx: %2d  case: %s  mean_Dice: %.4f  mean_HD95: %.4f  mean_IoU: %.4f  mean_Asd: %.4f' %
                     (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
                      np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))

    metric_list = metric_list / len(db_test)
    class_names = {
        1: "Aorta",
        2: "Gallbladder",
        3: "Kidney(L)",
        4: "Kidney(R)",
        5: "Liver",
        6: "Pancreas",
        7: "Spleen",
        8: "Stomach"
    }

    for i in range(1, args.num_classes):
        name = class_names[i]
        logging.info('Mean class: %-15s  Dice: %.4f  HD95: %.4f  IoU: %.4f  Asd: %.4f' %
                     (name, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance: mean_Dice : %.4f mean_HD95 : %.4f mean_IoU : %.4f mean_Asd : %.4f'
                 % (performance, mean_hd95, mean_iou, mean_asd))

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

    logging.basicConfig(filename=snapshot_path+'/'+"{}log.txt".format(args.epoch), level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    inference()


