#torch import 
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from torch.cuda import set_device


#Use this class as Template to make another Pytorch CNN pretrain model
class NAME_OF_CNN_HERE(nn.Module): #Name of CNN architecture. Rename to another architecture. Example ResNet
    def __init__(self,device = 'cuda:0'):
        super(NAME_OF_CNN_HERE, self).__init__()
        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        googlnet = models.CHANGE_PYTORCH_MODE_HERE(pretrained=True)
        set_device(device) #set device
        googlnet.float()
        googlnet.cuda()
        googlnet.eval()
        module_list = list(CHANGE_PYTORCH_MODEL_HERE.children())
        self.conv5 = nn.Sequential(*module_list[: -XXX_HERE])  # XXX_HERE number of layer you want

    def forward(self, x):
        x = self.transform(x) 
        x = x.unsqueeze(0)  
        x = Variable(x).cuda()
        feature = self.conv5(x)
        result = feature.view(feature.size(0), -1)

        return result
    """In this function must change. Example
        NAME_OF_CNN_HERE: GoogleNet, ResNet, or whatever you want.
        CHANGE_PYTORCH_MODEL_HERE: pytorch model check this link for more information https://pytorch.org/docs/stable/torchvision/models.html
        XXX_HERE: number of layer on CNN 
    """

#This class use to extract feature from a video. Inheritance class Feature
class ExtractFeatureVideo(Feature):
    def __init__():  
       pass
       #Some code here. No need to modify it.

       #......
       #,,,,,,

    
    def NAME_OF_CNN_HERE(self):

        self._method = 'name_of_cnn_here'
        self._framework = 'pytorch'
        dev = self._device.split(':')           
        if dev[0]=="GPU" or dev[0]=="gpu":
            self._device='cuda:'+dev[1]       
        elif dev[0]=='CPU' or dev[0]=="cpu":
            self._device='cpu:'+dev[1]
        else:
            self._device = self._device

        self.model = CHANGE_PYTORCH_MODEL_HERE(device=self._device)
        self.imageSize = (XXX_HERE,YYY_HERE) # XXX_HERE and YYY_HERE is size of input CNN above need
        return self._process()              
    """In this function must change. Example
        NAME_OF_CNN_HERE: GoogleNet, ResNet, or whatever you want.
        CHANGE_PYTORCH_MODEL_HERE: pytorch model check this link for more information https://pytorch.org/docs/stable/torchvision/models.html
        XXX_HERE and YYY_HERE: size of input image 
    """

    #Some code here. No need to modify it.

    #........
    #.....

#This class use to extract feature for all video from a dataset. Inheritance class ExtractFeatureVideo
class ExtractFeatureDataSet(ExtractFeatureVideo):

#END OF FILE