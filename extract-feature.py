"""
Code Feature Extract VGG-ResNet-Inception
Version: 0.3
Author: ThinhPLG - 29/04/2020
++++++++++++++++++++++++++++++
Log update:
    +Add class extract feature video: use for input is a video
    +Modify old class ExtractFeature to ExtractFeatureDataSet

    0.3 Update:
    + Add class for pytorch pretrain model
"""

import sys
import tensorflow as tf
import os
import numpy as np
import cv2 
from PIL import Image
from tqdm import tqdm
import logging
import time
#import datetime
import argparse

#keras import
from keras.preprocessing import image
from keras.applications import vgg16,vgg19,resnet,inception_v3,resnet_v2
from keras.models import Model
from keras.layers import Layer

#torch import 
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from torch.cuda import set_device


VIDEOSUM_FW_PATH ="/mmlabstorage/workingspace/VideoSum/videosummarizationframework/"
sys.path.append(os.path.join(VIDEOSUM_FW_PATH,'source/config')) #config path append
sys.path.append(os.path.join(VIDEOSUM_FW_PATH,'source/utilities'))
from config import cfg
from check_permission import check_permission_to_write
from parse_csv import get_metadata
from timerun import exec_time_func

#make log file

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=os.path.join(VIDEOSUM_FW_PATH,'source/log','feature-extract-log.txt'),
                    filemode='a')

#Use this class as Template to make another Pytorch CNN pretrain model
class GoogleNet(nn.Module):
    """GoogleNet(Inceptionv1) class
    Preprocess and compute feture result
    """    
    def __init__(self,device = 'cuda:0'):
        super(GoogleNet, self).__init__()
        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        googlnet = models.googlenet(pretrained=True)
        set_device(device) #set device
        googlnet.float()
        googlnet.cuda()
        googlnet.eval()
        module_list = list(googlnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])

    # rescale and normalize image, then pass it through GoogleNet
    def forward(self, x):
        x = self.transform(x) #Preprocess
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim
        x = Variable(x).cuda()
        res_conv5 = self.conv5(x)
        result = res_conv5.view(res_conv5.size(0), -1)

        return result

#Feature class
class Feature:
    def __init__(self,feat,name,method,sampling_rate):
        """Init function
        Arguments:
            feat {numpy 2d array} -- Array of feature vector
            name {str} -- Name of video
            method {str} -- Name of network use to extract
            sampling_rate {int} -- Sampling rate
        """        
        self._feature = feat
        self._sampling_rate=sampling_rate
        self._method = method
        self._namefile=name
        
    #If you want to save to another file extension such as: h5, zip. Try to write a function and add it below 
    def save(self,output_path):
        """Use to check premission before save data
        Arguments:
            output_path {str} -- folder path to save data
        """        
        #Write feature data to file
        if check_permission_to_write(output_path) is False:
            return
        try:
            self.__write_to_file_npy(output_path,self._namefile,self._feature)
        except:
            print("Error (save-Feature): try to extract feature before save")

    def __write_to_file_npy(self,output_path,name,data):
        """Use to write data to numpy file extension - `.NPY`

        Arguments:
            output_path {str} -- folder path to save data
            name {[type]} -- The name of file
            data {[type]} -- Data use to save
        """        
        path = os.path.join(output_path,
                            name+'_'+self._method+'_'+str(self._sampling_rate)+'.npy')
        np.save(path,data)
        print("Video: %s with sampling rate %d is save at %s"%(name,self._sampling_rate,path))

#This class use to extract feature from a video. Inheritance class Feature
class ExtractFeatureVideo(Feature):
    def __init__(self,path_video,sampling_rate=1,device_name='0'):  
        """Init function

        Arguments:
            Feature {Class} -- Inheritance Class
            path_video {str} -- Path to video

        Keyword Arguments:
            sampling_rate {int} -- Sampling rate (default: {1})
            device_name {str} -- Device use to extract (default: {'0'})
        """

        os.environ['CUDA_VISIBLE_DEVICES']=device_name #Set device envroment to use if system have multi device
        self._path = path_video
        self._namevideo = os.path.basename(path_video)          #Get basename from path
        self._namefile = os.path.splitext(self._namevideo)[0]   #Get name of video without extension file from path
        self._device = 'GPU:'+device_name                       #Return tensorflow form device GPU:x with x is index of GPU
        self._sampling_rate = sampling_rate
    def VGG19(self,output_layer='fc2'):
        """VGG19 - fc2 - output_shape = 4096. Tensorflow pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of VGG19 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """
        self._method = 'vgg19'
        self._framework = 'tensorflow'
        self._device = '/device:'+self._device          #Tensorflow device name
        base_model = vgg19.VGG19(weights='imagenet')    #Load pretrain model
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)    #Get layer weights
        _model.summary()             #show model architecture
        self.imageSize = (224,224)   #resize image
        self.model = _model
        self.preinput = vgg19.preprocess_input  #preprocess input function from keras.application
        return self._process()                  #return to _process function. Output of it is Feature Class

    def VGG16(self,output_layer='fc2'):
        """VGG16 - fc2 - output_shape = 4096. Tensorflow pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of VGG16 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """ 
        self._method = 'vgg16'
        self._framework = 'tensorflow'
        self._device = '/device:'+self._device                  #Tensorflow device name
        base_model = vgg16.VGG16(weights='imagenet')            #Load pretrain model
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)    #Get layer weights
        _model.summary()                        #show model architecture
        self.imageSize = (224,224)              #resize image
        self.model = _model
        self.preinput = vgg16.preprocess_input  #preprocess input function from keras.application
        return self._process()                   #return to _process function. Output of it is Feature Class

    def ResNet50(self,output_layer='avg_pool'):
        """ResNet50 - avg_pool - output_shape = 2048. Tensorflow pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of ResNet50 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """ 
        self.method = 'resnet50'
        self._framework = 'tensorflow'
        self._device = '/device:'+self._device
        base_model = resnet.ResNet50(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = resnet.preprocess_input
        return self._process()

    def ResNet152(self,output_layer='avg_pool'):
        """ResNet152 - avg_pool - output_shape = 2048. Tensorflow pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of ResNet50 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """ 
        self._method = 'resnet152'
        self._framework = 'tensorflow'
        self._device = '/device:'+self._device
        base_model = resnet.ResNet152(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = resnet.preprocess_input
        return self._process()

    def InceptionV3(self,output_layer='avg_pool'):
        """InceptionV3 - avg_pool - output_shape = 2048. Tensorflow pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of InceptionV3 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """ 
        self._method = 'inceptionv3'
        self._framework = 'tensorflow'
        #self._device = 'device:'+self._device
        base_model = inception_v3.InceptionV3(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (299,299)
        self.model = _model
        self.preinput = inception_v3.preprocess_input
        return self._process()
    
    def InceptionV1(self):
        """InceptionV1 - avg_pool - output_shape = 2048. Pytorch pretrain model.

        Keyword Arguments:
            output_layer {str} -- The layer of InceptionV1 architecture (default: {'fc2'})

        Returns:
            [Feature Object] -- Return Feature Object from function _process. 
        """ 
        self._method = 'inceptionv1'
        self._framework = 'pytorch'
        dev = self._device.split(':')           #split device name to get GPU or CPU and index of it
        if dev[0]=="GPU" or dev[0]=="gpu":
            self._device='cuda:'+dev[1]         #Set dev to form: cuda:x with x is index
        elif dev[0]=='CPU' or dev[0]=="cpu":
            self._device='cpu:'+dev[1]          #Set dev to form: cpu:x with x is index
        else:
            self._device = self._device         ##Set dev to it name if you on anther envroment

        self.model = GoogleNet(device=self._device) #Get class Googlenet above
        self.imageSize = (224,224)          #size of input image use to resize function
        return self._process()              

    def _process(self):
        if self._framework == 'tensorflow': #check if framework is tensorflow and call tensorflow function
            #tf.debugging.set_log_device_placement(True)
            #try:
            #    with tf.device(self._device):
                    #print(self._device)
            feat = self.__process_video_tensorflow()
            #except RuntimeError as e:
            #    logging.error(e)
        elif self._framework == 'pytorch':  #check if framework is pytorch and call tensorflow function
            feat = self.__process_video_pytorch()
        return Feature(feat,self._namefile,self.method,self._sampling_rate)  #Feature class

    def __process_video_pytorch(self):
        """Process video if it is pytorch pretrain model

        Returns:
            [Feauture Class] -- Feature Class use to save or something else
        """        
        print("Extracting %s with sampling rate is %d on %s"%(self._namevideo,self._sampling_rate,self._method))
        vidcap = cv2.VideoCapture(self._path)       #open video
        if (vidcap.isOpened()== False):             #check opened
            #check opened?
            logging.error("Fail to open video %s"%(video))
        sam = self._sampling_rate               #Set sampling rate
        nFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)       #Get total frames
        pbar = tqdm(total = nFrame)
        it = 0
        feature = []
        while(vidcap.isOpened()):
            pbar.update(1)
            suc, img = vidcap.read()        #read image
            it+=1
            if(suc == False):               #Break while loop if cant read frame image
                break
            if ((it-1)%sam) != 0:           #Sampling
                continue
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #convert RGB
            frame = cv2.resize(frame, self.imageSize)           #Resize image
            res_pool5 = self.model(frame)                       #extract from model
            frame_feat = res_pool5.cpu().data.numpy().flatten()     #Convert from tensor(pytorch) to numpy array
            feature.append(frame_feat) 

        return Feature(feature,self._namefile,self.method,self._sampling_rate)

    def __process_video_tensorflow(self):
        """Process video if it is pytorch pretrain model

        Returns:
            [Feauture Class] -- Feature Class use to save or something else
        """

        print("Extracting %s with sampling rate is %d on %s"%(self._namevideo,self._sampling_rate,self._method))
        vidcap = cv2.VideoCapture(self._path)       #open video
        if (vidcap.isOpened()== False):             #Check opened
            #check opened?
            logging.error("Fail to open video %s"%(video))
        sam = self._sampling_rate
        nFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)   #get total Frames
        pbar = tqdm(total = nFrame)
        it = 0
        feat = []
        while(vidcap.isOpened()):
            pbar.update(1)
            suc, img = vidcap.read()            #Read image frames
            it+=1
            if(suc == False):                   #Break while loop if cant read frame image
                break
            if ((it-1)%sam) != 0:               #Sampling
                continue
            _feature = self.__extract(img)
            feat.append(_feature)
        res = np.asarray(feat)              #convert to numpy array
        #self.feature = np.squeeze(res)
        res = np.squeeze(res)
        return res

    def __extract(self,input_image):
        """This function use to extract feature for tensorflow framework

        Arguments:
            input_image {cv2 image} -- image use to extract

        Returns:
            Freture -- Feature vector
        """        
        #Extract feature from image                    
        img = Image.fromarray(input_image)          #convert Image type
        img = img.resize(self.imageSize)            #resize
        img_data = image.img_to_array(img)          
        img_data = np.expand_dims(img_data, axis=0)
        img_data = self.preinput(img_data)          #Preprocess
        _feature = self.model.predict(img_data)     #Get feauter vector
        return _feature

#This class use to extract feature for all video from a dataset. Inheritance class ExtractFeatureVideo
class ExtractFeatureDataSet(ExtractFeatureVideo):
    def __init__(self,dataset_name,output_path,from_id=None,to_id=None,sampling_rate=1,device_name='0'):
        """Init function

        Arguments:
            ExtractFeatureVideo {Class } -- Inheritance class
            dataset_name {str} -- The name of dataset
            output_path {str} -- Folder path to save feature

        Keyword Arguments:
            from_id {int} -- From video x on dataset. (default: {None})
            to_id {int} -- To video y on dataset. y>=x (default: {None})
            sampling_rate {int} -- Sampling rate (default: {1})
            device_name {str} -- Name of device (default: {'0'})
        """ 
        #super().__init__(sampling_rate=sampling_rate,device_name=device_name)
        os.environ['CUDA_VISIBLE_DEVICES']=device_name      #Set device envroment to use if system have multi device
        self._dataset = dataset_name         
        self._output = output_path
        self._device = 'GPU:'+device_name           #Return tensorflow form device GPU:x with x is index of GPU
        self._x = from_id                   #Set x and y. Use to choose video from video x to video y of dataset
        self._y = to_id
        self._samplingRate = sampling_rate
        if check_permission_to_write(self._output) is False:       #Check permisstion first
            sys.exit()

    def _process(self):
        #overwrite function process of class ExtractFeatureVideo
        if self._framework is 'tensorflow':
            #try:
            #    with tf.device(self._device):
            feat = self.__process_dataset_tensorflow()
            #except RuntimeError as e:
            #    logging.error(e)
        elif self._framework is 'pytorch':
            feat = self.__process_dataset_pytorch()
        return Feature(feat,self._namefile,self.method,self._sampling_rate)

    def __process_dataset_pytorch(self):
        #overwrite function process of class ExtractFeatureVideo
        x = self._x
        if x is None:    #check x is None and set x=0
            x=0
        videoName,videoPathLists,_,nFrame,_ = self._read_meta_data()    #Get meta data of dataset [VideoName,VideoPath,Fps,Total frame,Duration]
        for idx,video in enumerate(videoName):
            feature = []
            print("%s/%s : %s with sampling rate is %d"%(idx+1,len(videoName),video,self._samplingRate))
            path = videoPathLists[idx+x]        #Plus x to jump to video x-th on dataset. Check pandas.read_csv. 
            vidcap = cv2.VideoCapture(path)
            if (vidcap.isOpened()== False):     #Check opened
                #check opened?
                logging.error("Fail to open video %s"%(video))
            sam = self._samplingRate
            pbar = tqdm(total = nFrame[idx+x])
            it = 0
            while(vidcap.isOpened()):       
                pbar.update(1)
                suc, img = vidcap.read()        #read image
                it+=1
                if(suc == False):               #Break while loop if cant read frame image
                    break
                if ((it-1)%sam) != 0:
                    continue
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #convert RGB
                frame = cv2.resize(frame, self.imageSize)       #Resize image
                res_pool5 = self.model(frame)                   #Extract from model
                frame_feat = res_pool5.cpu().data.numpy().flatten()     #Convert from tensor(pytorch) to numpy array
                feature.append(frame_feat)
                    
            namefile = os.path.splitext(os.path.basename(video))[0]
            self._write_to_file(namefile,feature)               #write to file

    def __process_dataset_tensorflow(self):
        #overwrite function process of class ExtractFeatureVideo
        x = self._x
        if x is None:
            x=0
        videoName,videoPathLists,_,nFrame,_ = self._read_meta_data()        #Get meta data of dataset [VideoName,VideoPath,Fps,Total frame,Duration]
        for idx,video in enumerate(videoName):
            feat = []
            print("%s/%s : %s with sampling rate is %d"%(idx+1,len(videoName),video,self._samplingRate))
            path = videoPathLists[idx+x]        #Plus x to jump to video x-th on dataset. Check pandas.read_csv. 
            vidcap = cv2.VideoCapture(path)
            if (vidcap.isOpened()== False):             
                #check opened?
                logging.error("Fail to open video %s"%(video))
            sam = self._samplingRate
            pbar = tqdm(total = nFrame[idx+x])
            it = 0
            while(vidcap.isOpened()):           
                pbar.update(1)
                suc, img = vidcap.read()
                it+=1
                if(suc == False):                   #Break while loop if cant read frame image
                    break
                if ((it-1)%sam) != 0:
                    continue
                _feature = self.__extract(img)      #extract 
                feat.append(_feature)
            result = np.asarray(feat)
            result = np.squeeze(result)
            namefile = os.path.splitext(os.path.basename(video))[0]
            self._write_to_file(namefile,result)

    def _read_meta_data(self):
        """Read data from file csv

        Returns:
            Namevid {Array} -- Name of each video on dataset
            Path {Array} -- Path to each video on dataset
            Fps {Array} -- Fps of each video on dataset
            nFrames {Array} -- Total frame of each video on dataset
            Duaration {Array} -- Duaration of each video on dataset
        """                
        dn = self._dataset   #dataset name
        if dn == 'bbc' or dn =='BBC' or dn == 'BBC EastEnders':
            namevid,path,fps,nFrame,duration = get_metadata('bbc')
        elif dn == 'summe' or dn =='SumMe' or dn == 'SUMME':
            namevid,path,fps,nFrame,duration = get_metadata('summe')
        else:
            namevid,path,fps,nFrame,duration = get_metadata(dn)
        x = self._x
        y = self._y
        return namevid[x:y],path[x:y],fps[x:y],nFrame[x:y],duration[x:y]

    def _write_to_file(self,name,data):
        #Write feature data to file
        self.__write_to_file_npy(name,data)

    def __write_to_file_npy(self,name,data):
        """Write to file numpy extension (npy)

        Arguments:
            name {str} -- Name of video
            data {numpy array} -- data use to write
        """        
        path = os.path.join(self._output,
                            name+'_'+self._method+'_'+str(self._samplingRate)+'.npy')
        np.save(path,data)
        logging.info("Dataset: %s Video: %s with sampling rate %d is save at %s"%(self._dataset,name,self._samplingRate,path))

    def __extract(self,input_image):
        #overwrite function process of class ExtractFeatureVideo
        #Extract feature from image                
        img = Image.fromarray(input_image)
        img = img.resize(self.imageSize)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = self.preinput(img_data)
        _feature = self.model.predict(img_data)
        return _feature

def args_define():
    """Fuction to define arguments
    """    
    parser = argparse.ArgumentParser(description='EXTRACT FEATURE TOOLs')

    parser.add_argument(
        'Method', action='store',type=str, help='Method use to extract. Example: Inceptionv1, ResNet50, ResNet151, VGG16, VGG19')
    parser.add_argument(
        'InputType', action='store',type=str, help='Choose input type:Video or Dataset. Example: video, v, V, Dataset')    
    subparsers = parser.add_subparsers()


    video = subparsers.add_parser(
        '+Video', help='Extract Feature for a VIDEO')
    video.add_argument(
        'VideoPath', action='store',type=str, help='Path to video')
    video.add_argument(
        '-sr','--samplingRate', action='store',default=1,type=int, help='Sampling Rate')
    video.add_argument(
        '-d','--device', action='store',type=str,default='0', help='Device use to run. ID of GPUS')
    video.add_argument(
        '-l','--layer', action='store',type=str, help='Layer of CNN. (Optional: Only support Keras model)')
    video.add_argument(
        '-s','--save', action='store',type=str, help='Path to save result')
    

    dataset = subparsers.add_parser(
        '+DataSet', help='Extract Feature for a DataSet')
    dataset.add_argument(
        'DataSetName', action='store',type=str, help='Name of dataset')
    dataset.add_argument(
        'OutputPath', action='store',type=str, help='Folder output path')
    dataset.add_argument(
        '-f','--fromid', action='store',type=int,default=None, help='Choose starting order of video want to run')
    dataset.add_argument(
        '-e','--endid', action='store',type=int,default=None, help='Choose ending order of video want to run')
    dataset.add_argument(
        '-sr','--samplingRate', action='store',default=1,type=int, help='Sampling Rate')
    dataset.add_argument(
        '-d','--device', action='store',type=str,default='0', help='Device use to run')
    dataset.add_argument(
        '-l','--layer', action='store',type=str,default=None, help='Layer of CNN. (Optional: Only support Keras model)')
    args = parser.parse_args()
    return args

def choose_method(module,name_method,lay):
    """This function use to get feature array from module and name method.

    Arguments:
        module {ExtractFeatureDataSet Object} -- ExtractFeatureDataSet init
        name_method {str} -- The name of method CNN use to extract
        lay {str} -- Name layer

    Returns:
        Feature Class -- Feature
    """    
    if name_method == "inceptionv1": #or "InceptionV1" or "googlenet":
        if lay is not None:
            print('InceptionV1 is pytorch model. Not support to choose layer')
        return module.InceptionV1()
    elif name_method == 'inceptionv3':# or 'InceptionV3' or 'Inceptionv3':
        if lay is not None:
            return module.InceptionV3(output_layer=lay)
        else:
            return module.InceptionV3()
    elif name_method == 'resnet50':# or 'ResNet50' or 'Resnet50':
        if lay is not None:
            return module.ResNet50(output_layer=lay)
        else:
            return module.ResNet50()
    elif name_method == 'resnet152':# or 'ResNet152' or 'Resnet152':
        if lay is not None:
            return module.ResNet152(output_layer=lay)
        else:
            return module.ResNet152()
    elif name_method == 'vgg16':# or 'Vgg16' or 'vgg16':
        if lay is not None:
            return module.VGG16(output_layer=lay)
        else:
            return module.VGG16()
    elif name_method == 'vgg19':# or 'Vgg19' or 'vgg19':
        if lay is not None:
            return module.VGG19(output_layer=lay)
        else:
            return module.VGG19()
    else:
        print("Error on function `choose_method`: Check name method again!!!")
def main():
    args = args_define()
    print(args)
    if args.InputType == 'v' or args.InputType == 'V' or args.InputType == 'Video':
        try:
            if args.VideoPath is not None:
                module = ExtractFeatureVideo(args.VideoPath,sampling_rate=args.samplingRate,device_name=args.device)
                feature = choose_method(module,args.Method,args.layer)
                if args.save is not None:
                    feature.save(args.save)
        except NameError as e:
            print(e,'Error: Must to input VideoPath')
    else:
        try:
            print(args.DataSetName,args.OutputPath)
            if args.DataSetName != '':#and args.OutputPath != None:
                module = ExtractFeatureDataSet(args.DataSetName,args.OutputPath,from_id=args.fromid,to_id=args.endid,sampling_rate=args.samplingRate,device_name=args.device)
                choose_method(module,args.Method,args.layer)
                print("da co dataset")
        except NameError as e:
            print(e,'Error: Must to input DataSetName and OutputPath')

if __name__ == '__main__':
    main()