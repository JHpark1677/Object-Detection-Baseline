import configargparse


def get_args_parser():
    # * config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument("--path", default="D:\data", type=str,help='data path')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch_num", default=300, type=int)

    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--resume",'-r',action='store_true', help='resume from checkpoint')
    parser.add_argument('--load_ckp',default='ckpt_vit.pth',type=str, help='checkpoint_name')
    parser.add_argument('--save_ckp',default='ckpt_vit.pth',type=str, help='checkpoint_name')

    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    parser.add_argument('--distributed_true', dest='distributed', action='store_true')
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])   # usage : --gpu_ids 0, 1, 2, 3
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int)
    
    return parser