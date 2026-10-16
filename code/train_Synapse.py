import argparse
import logging
import os
import random
import sys
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataset_Synapse import (Synapse_dataset, RandomGenerator)
from utils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss

from redino.NIX_Synapse import Dino_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse', help='Dataset root path')
parser.add_argument('--exp', type=str, default='Synapse', help='Name of Experiment')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch size per gpu')
parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224], help='input patch size of network input')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--seed', type=int, default=1314, help='random seed')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
args = parser.parse_args()


def train(snapshot_path):
    model = Dino_seg(num_classes=args.num_classes).cuda()
    db_train = Synapse_dataset(base_dir=args.root_path, split="train", transform=RandomGenerator(args.input_size))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8,
                             worker_init_fn=worker_init_fn)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    writer = SummaryWriter(snapshot_path + '/log')

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    base_lrs = [group['lr'] for group in optimizer.param_groups]
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("iterations per epoch:{}    max iterations:{}".format(len(trainloader), max_iterations))
    epoches = tqdm(range(max_epoch), ncols=70)

    for epoch_num in epoches:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            output = model(image_batch)

            loss_ce = ce_loss(output, label_batch[:].long())
            loss_dice = dice_loss(output, label_batch, softmax=True)
            loss = 0.3 * loss_ce + 0.7 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_factor = (1.0 - iter_num / max_iterations) ** 0.9
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = base_lrs[i] * lr_factor

            iter_num = iter_num + 1
            writer.add_scalar('loss/total_loss', loss, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            logging.info(
                'iteration %d : loss: %f   loss_ce: %f   loss_dice: %f' % (iter_num, loss.item(), loss_ce.item()
                                                                           , loss_dice.item()))

            if iter_num % 50 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_epochs = [max_epoch - 1 - i * 5 for i in range(10)]
        if epoch_num in save_epochs:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
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

    snapshot_path = "../model/{}".format(args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(snapshot_path)
