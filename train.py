
import os
import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.keep_cutout import Cutout, Keep_Cutout, Keep_Cutout_Low
from util.keep_autoaugment import CIFAR10Policy, Keep_Autoaugment, Keep_Autoaugment_Low
from model.resnet import ResNet18, ResNet18_Early
from model.wide_resnet import WideResNet, WideResNet_Early
from model.shake_resnet import ShakeResNet, ShakeResNet_Early


model_options = ['resnet', 'wideresnet', 'shake']
dataset_options = ['cifar10']
method_options = ['none', 'cutout', 'keep_cutout', 'keep_cutout_low', 'keep_cutout_early', 'keep_cutout_low_early',
                   'autoaugment','keep_autoaugment','keep_autoaugment_low', 'keep_autoaugment_early', 'keep_autoaugment_low_early']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet', choices=model_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--length', type=int, default=16, help='length of the holes')
parser.add_argument('--N', type=int, default=3, help='number of autoaugments')
parser.add_argument('--M', type=int, default=24, help='magnitude of autoaugments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
parser.add_argument('--method', default='none', choices=method_options)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model + '_' + args.method

print(args)

# Image Preprocessing
mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
normalize = transforms.Normalize(mean, std)


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.method=='cutout':
    train_transform.transforms.append(Cutout(n_holes=1, length=args.length))
elif args.method=='autoaugment':
    train_transform.transforms.insert(2, CIFAR10Policy())
elif args.method=='keep_cutout':
    keep = Keep_Cutout(train_transform, mean, std, args.length)
elif args.method=='keep_cutout_low':
    keep = Keep_Cutout_Low(train_transform, mean, std, args.length)
elif args.method=='keep_cutout_early':
    keep = Keep_Cutout(train_transform, mean, std, args.length, True)
elif args.method=='keep_cutout_low_early':
    keep = Keep_Cutout_Low(train_transform, mean, std, args.length, True)
elif args.method=='keep_autoaugment':
    keep = Keep_Autoaugment(train_transform, mean, std, args.length, args.N, args.M)
elif args.method=='keep_autoaugment_low':
    keep = Keep_Autoaugment_Low(train_transform, mean, std, args.length, args.N, args.M)
elif args.method=='keep_autoaugment_early':
    keep = Keep_Autoaugment(train_transform, mean, std, args.length, args.N, args.M, True)
elif args.method=='keep_autoaugment_low_early':
    keep = Keep_Autoaugment_Low(train_transform, mean, std, args.length, args.N, args.M, True)



train_dataset = datasets.CIFAR10(root='data/',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=8)

if args.model == 'shake':
    cnn = ShakeResNet(depth=26, w_base=64, label=10)
    if 'early' in args.method:
        cnn = ShakeResNet_Early(depth=26, w_base=64, label=10)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28,widen_factor=10, dropout_rate=0.0, num_classes=10)
    if 'early' in args.method:
        cnn = WideResNet_Early(depth=28,widen_factor=10, dropout_rate=0.0, num_classes=10)
else:
    cnn = ResNet18(num_classes=10)
    if 'early' in args.method:
        cnn = ResNet18_Early(num_classes=10)
        
print(cnn)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = CosineAnnealingLR(cnn_optimizer, T_max=args.epochs, eta_min=0.)

if not os.path.isdir('logs'):
    os.mkdir('logs')

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            if 'early' in args.method:  
                pred,_ = cnn(images,False)
            else:
                pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()
        if 'keep' in args.method: 
            images = keep(images, cnn)

            
        cnn.zero_grad()
        if 'early' in args.method:  
            pred,aux_pred = cnn(images,False)
            xentropy_loss = criterion(pred, labels)
            aux_loss = criterion(aux_pred, labels)
            xentropy_loss += aux_loss*0.3
        else:
            pred = cnn(images)
            xentropy_loss = criterion(pred, labels)
            
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)  # Use this line for PyTorch <1.4
    # scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
