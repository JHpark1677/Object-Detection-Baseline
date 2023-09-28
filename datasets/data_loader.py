import os
import torch
from torch.utils.data.distributed import DistributedSampler
from voc_dataloader import YOLO_PASCAL_VOC

def dataloader(args):
    if args.dataset == 'voc':
        
        path2data = 'C:/Users/RTL/Documents/GitHub/PyTorch-FusionStudio'
        if not os.path.exists(path2data):
            os.mkdir(path2data)

        trainset_1 = YOLO_PASCAL_VOC(path2data, year='2007', image_set='trainval', download=True)
        trainset_2 = YOLO_PASCAL_VOC(path2data, year='2012', image_set='trainval', download=True)
        testset_1 = YOLO_PASCAL_VOC(path2data, year='2007', image_set='test', download=True)
        testset_2 = YOLO_PASCAL_VOC(path2data, year='2012', image_set='test', download=True)

        trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2])
        testset = torch.utils.data.ConcatDataset([testset_1, testset_2])

        if args.distributed:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size/args.world_size), 
                                                      num_workers=int(args.num_workers/args.world_size), 
                                                      pin_memory=True, sampler=DistributedSampler(dataset=trainset), drop_last=True
                                                      )
            testloader = torch.utils.data.DataLoader(testset,
                                     batch_size=int(args.batch_size/args.world_size),
                                     num_workers=int(args.num_workers/args.world_size),
                                     pin_memory=True,
                                     sampler=DistributedSampler(dataset=testset, shuffle=False),
                                     drop_last=False)
            
    return trainloader, testloader