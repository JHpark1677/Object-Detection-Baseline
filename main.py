import os
import torch

import visdom
from torchvision import models
import torch.optim as optim
import models
from datasets import data_loader

from util import loss
from utils import resume
from utils import init_for_distributed
import eval
import train

def main_worker(rank, args):
    # 1. Init distributed
    if args.distributed:
        init_for_distributed(rank, args)

    # 2. device
    device = torch.device('cuda:{}'.format(int(args.gpu_ids[args.rank])))
    print("My Device :", device)

    # 3. visdom
    if args.visdom_true:
        vis = visdom.Visdom()
    else:
        vis = None
    
    # 4. data loader
    path2data = 'D:\data'
    if not os.path.exists(path2data):
        os.mkdir(path2data)

    trainloader, testloader = data_loader.dataloader(args)

    # 5. model load
    if args.resume :
        model = models.yolo_v1_impl_3.Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
        model.state_dict(resume.load_ckp(args))
    else :
        model = models.yolo_v1_impl_3.Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    
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
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.00002, weight_decay=5e-4)
    criterion = loss.YoloLoss()
    args.epoch_num = 300

    #6. train and eval
    start_epoch = 0

    for epoch in range(start_epoch+1, start_epoch+args.epoch_num+1):
        
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
            
        train.train(model, trainloader, optimizer, criterion, epoch, args, vis, device)
        eval.evaluate(model, testloader, epoch, args, vis, device)


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