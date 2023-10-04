import torch

def vis_loss_report(vis, args, epoch, mean_loss):
    if vis is not None:
        # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                    Y=torch.Tensor([sum(mean_loss)/len(mean_loss)]).unsqueeze(0).cpu(),
                    win='train_loss_' + args.save_ckp,
                    update='append',
                    opts=dict(xlabel='step',
                            ylabel='Loss',
                            title='train_loss_{}'.format(args.save_ckp),
                            legend=['Total Loss']))
        
        
def vis_map_report(vis, args, epoch, mean_loss, mean_avg_prec, stage):
    if stage == "Test" :
        if args.rank == 0:
            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                            Y=torch.Tensor([mean_avg_prec]).unsqueeze(0).cpu(),
                            win='{}_map_'.format(stage) + args.save_ckp,
                            update='append',
                            opts=dict(xlabel='epoch',
                                    ylabel='{}_Loss'.format(stage),
                                    title='Total_Loss_{}'.format(args.save_ckp),
                                    legend=['{} mAP'.format(stage)]))
    else :        
        if args.rank == 0:
            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                            Y=torch.Tensor([mean_loss, mean_avg_prec]).unsqueeze(0).cpu(),
                            win='{}_loss_map'.format(stage) + args.save_ckp,
                            update='append',
                            opts=dict(xlabel='epoch',
                                    ylabel='{}_Loss'.format(stage),
                                    title='Total_Loss_{}'.format(args.save_ckp),
                                    legend=['{} Loss', 'mAP'.format(stage)]))