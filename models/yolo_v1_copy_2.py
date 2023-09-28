import torch
import torch.nn as nn

class yolo_v1(nn.Module):
    def __init__(self, num_classes=91, num_bboxes=2):
        super(yolo_v1, self).__init__()
        
        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4),
            nn.LeakyReLU(0.1, inplace=True), # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. Memory Usage가 좋아지고, input을 없앰. 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, strid=1, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0), # 왜 갑자기 channnel_size가 줄어드는 것이지?
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channles=128, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0), 
            nn.LeakyReLU(0.1, inplace=True), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fc = nn.Sequential(
            Flatten(), 
            nn.Linear(in_features=7*7*1024, out_features=4096), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=4096, out_features=(self.feature_size*self.feature_size*(5*num_bboxes+num_classes))), 
            nn.Softmax()
        )
        self.init_weight(self.conv)
        self.init_weight(self.fc)
    
    def forward(self, x):
        s, b, c = self.feature_size, self.num_bboxes, self.num_classes

        x = self.conv(x)
        x = self.fc(x)

        x = x.view(-1, s, s, (5*b+c))
        return x
    
    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
