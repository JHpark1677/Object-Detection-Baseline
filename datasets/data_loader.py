import os

import torch
from torch.utils.data.distributed import DistributedSampler
from .voc_dataloader import YOLO_PASCAL_VOC

def dataloader(args):

    if args.dataset == 'voc':
        
        path2data = 'C:/Users/RTL/Documents/GitHub/PyTorch-FusionStudio'
        if not os.path.exists(path2data):
            os.mkdir(path2data)

        trainset_1 = YOLO_PASCAL_VOC(path2data, year='2007', image_set='train', download=True)
        trainset_2 = YOLO_PASCAL_VOC(path2data, year='2012', image_set='train', download=True)
        testset_1 = YOLO_PASCAL_VOC(path2data, year='2007', image_set='val', download=True)
        testset_2 = YOLO_PASCAL_VOC(path2data, year='2012', image_set='val', download=True)

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
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=2)
    print("train set size :", args.batch_size * len(trainloader))
    print("test set size :", args.batch_size * len(testloader))
    return trainloader, testloader


# if __name__ == '__main__':
#     import torch.multiprocessing as mp
#     from config import get_args_parser
#     import configargparse

#     parser = configargparse.ArgumentParser('ResNet', parents=[get_args_parser()])
#     args = parser.parse_args()
#     trainloader, testloader = dataloader(args)
#     print("length of trainloader : ", len(trainloader))
#     print("length of testloader : ", len(testloader))