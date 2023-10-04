import torch
import os

def load_ckp(args):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
    path = '../checkpoint/' + os.path.join(args.load_ckp)
    model = torch.load(path)
    return model

def save_ckp(model, args):
    print("==> Saving checkpoint")
    assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
    path = '../checkpoint/' + os.path.join(args.save_ckp)
    torch.save(model, path)