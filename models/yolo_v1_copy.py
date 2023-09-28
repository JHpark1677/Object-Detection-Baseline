import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import resnet50

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    # register_buffer로 layer를 등록하게 되면 -> optimizer update가 일어나지 않고, 값이 존재하며, state_dict()로 확인 가능하다.
    # end2end 학습을 시키는데, 중간에 update하지 않는 일반 layer를 넣고 싶을 때 사용할 수 있다. 

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class yolo(nn.Module):
    def __init__(self):
        super(yolo, self).__init__()
        self.backbone = nn.Sequential(*list(nn.ModuleList(resnet50(pretrained=True,
                                                                   norm_layer=FrozenBatchNorm2d).children()))[:-2]) # 마지막 2개 Layer를 제외한 나머지를 가져온다.
        
        for name, parameter in self.backbone.named_parameters():
            if '5.' not in name and '6.' not in name and '7.' not in name:
                parameter.requires_grad_(False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1), 
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1), 
            nn.LeakyReLU(), 
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(7*7*1024, 4096), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(4096, 1470) # 7 * 7 * 30
        )

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d) :
                nn.init.normal_(m.weight, mean=0, std=0.01)

        for m in self.linear.modules():
            if isinstance(m, nn.Linear) :
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        out = self.backbone(x)
        print("what about this? : ", out.shape)
        out = self.conv(out)
        out = self.linear(out)
        out = torch.reshape(out, (-1, 7, 7, 30))
        
        return out

def test():
    net = yolo()
    batch_size = 16
    x = torch.rand(batch_size, 3, 800, 800)
    y = net(x)
    print(y)

if __name__ == '__main__':
    test()