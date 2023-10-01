import os
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import models
from datasets import data_loader

from util import loss_copy_1
from util import utils
from utils import init_for_distributed
import eval
import train

def main_worker(rank, args):

    if args.distributed:
        init_for_distributed(rank, args)

    # 2. device
    device = torch.device('cuda:{}'.format(int(args.gpu_ids[args.rank])))
    print("My Device :", device)

    # 3. data loader
    path2data = 'D:\data'
    if not os.path.exists(path2data):
        os.mkdir(path2data)

    trainloader, testloader = data_loader.dataloader(args)

    # 4. model load

    model = models.yolo_v1_copy_3.Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        path = '../checkpoint/' + os.path.join(args.load_ckp)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[int(args.gpu_ids[args.rank])], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    # 5. optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.00001, momentum = 0.9, weight_decay=5e-4)
    criterion = loss_copy_1.YoloLoss()
    args.epoch_num = 300
    test_accuracy = 0

    #6. train and eval
    start_epoch = 0
    best_acc = 0

    for epoch in range(start_epoch+1, start_epoch+args.epoch_num+1):
        
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
            
        train.train(model, trainloader, optimizer, criterion, epoch, args, device)
        #test_loss, test_accuracy = eval.evaluate(model, testloader, criterion, device)


if __name__ == "__main__": 
    import torch.multiprocessing as mp
    from config import get_args_parser
    import configargparse

    parser = configargparse.ArgumentParser('YOLO', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if len(args.gpu_ids) > 1:
        args.distributed = True
    else:
        args.distributed = False

    args.world_size = len(args.gpu_ids)
    args.num_workers = len(args.gpu_ids) * 4

    if args.distributed:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(args.rank, args)