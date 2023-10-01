import torch
import os

def load_ckp(model, args):
    if args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
            path = '../checkpoint/' + os.path.join(args.load_ckp)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])