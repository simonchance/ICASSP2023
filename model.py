import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
#from torchsummary import summary
import collections
from torchvision import models
from unet_parts import *

class Node(nn.Module):
    def __init__(self, in_dim, out_dim, node_kernel = 1):

        super(Node, self).__init__()
        self.node = nn.Sequential(
            nn.Conv2d(in_dim,out_dim,kernel_size=node_kernel, stride=1,padding=node_kernel // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))
        #k = 3?
    def forward(self, x):
        return self.node(x)

class BranchNet(nn.Module):
    def __init__(self,in_channels =128 ,out_channels = 128):
        super(BranchNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.chattention = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding =1, dilation=1) 
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding =2, dilation=2)
        self.conv3 = nn.Conv2d(384, 128, kernel_size=3, padding =3, dilation=3)

        self.convd1 = nn.Conv2d(640, 128, kernel_size=1)  # 640
        self.convd2 = nn.Conv2d(640, 128, kernel_size=1)  # 640


    def forward(self, x):
       
        f1 = self.relu(self.conv1(x))
        f1 = torch.cat((f1,x),dim=1)

        f2 = self.relu(self.conv2(f1))
        f2 = torch.cat((f2,f1),dim=1)

        f3 = self.relu(self.conv3(f2))#conv2?f1?
        f3 = torch.cat((f3,f2),dim=1)
        
        fa = self.relu(self.convd1(torch.cat((f1,f2),dim=1)))
        fb = torch.cat((fa,f3),dim=1)
        f4 = self.relu(self.convd2(fb))
        
        f4 = self.relu(self.chattention(self.pool(f4)))*f4
        f5 = f4+x
        return f5

class GBSNet(nn.Module):

    def __init__(self, load_weights=False):

        super(GBSNet, self).__init__()
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,512]
        self.convdown = nn.Conv2d(512, 128, kernel_size=1)
        self.branch1 = BranchNet(in_channels = 128,out_channels = 128)
        self.branch2 = BranchNet(in_channels = 128,out_channels = 128)
        self.branch3 = BranchNet(in_channels = 128,out_channels = 128)
        self.branch4 = BranchNet(in_channels = 128,out_channels = 128)
        self.branch5 = BranchNet(in_channels = 128,out_channels = 128)

        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
        self.relu = nn.ReLU()
        
        self.features = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64, 2, stride=2, padding=0, output_padding=0, groups = 1, bias = False),

            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32, 2, stride=2, padding=0, output_padding=0, groups = 1, bias = False),

            nn.Conv2d(32, 32, kernel_size=3, padding=3, dilation = 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32,32, 2, stride=2, padding=0, output_padding=0, groups = 1, bias = False),

        )
        self.relu = nn.ReLU(inplace=True)
        self.frontend = make_layers(self.frontend_feat)
        
        #权重初始化
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):#10个卷积*（weight，bias）=20个参数
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

        #------- init weights --------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #-----------------------------#

    def forward(self, input):
        outputs = []
        
        fv = self.frontend(input)
        
        #fv = self.convdown(fv)
        fv = self.convdown(fv)

        fv1 = self.branch1(fv)
        #fv11 = fv1+fv
        fv2 = self.branch2(fv1)
        #fv22 = fv2+fv11
        fv3 = self.branch3(fv2)
        #fv33 = fv3+fv22
        fv4 = self.branch4(fv3)
        #fv44 = fv4+fv33
        fv5 = self.branch5(fv4)


        fvf = fv5+fv
        outputs.append(fvf)
        x = self.features(fvf)
        x = self.output_layer(x)     
        outputs.append(x)
        return outputs
        
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):  #dialtion扩张
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    
if __name__=="__main__":
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GBSNet().to(device)
    input_img=torch.ones((1,3,32,32)).to('cuda')
    out=model(input_img)
    print(out.shape)
    torch.save(model, 'net1.pkl')
    #summary(model, (3,640,480))   
   
    
