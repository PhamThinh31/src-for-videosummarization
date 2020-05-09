"""
Code Feature Extract VGG-ResNet-Inception-FisherVector 
    #This code support 3 dataset: Summe, Tvsum50, BBC. 
    #To change data set: make a file csv with 5 fields {videname,path,fps,total frames,duration}
    #See example on data/input_list/TVSum.csv
Version: 0.1
Author: ThinhPLG - 29/04/2020
++++++++++++++++++++++++++++++
Todo: +fuction write to h5 file
      +fuction input from frame video
"""
import sys
import tensorflow as tf
import os
import numpy as np
import cv2
import logging 
from PIL import Image
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications import vgg16,vgg19,resnet,inception_v3,resnet_v2
from keras.models import Model
from keras.layers import Layer

VIDEOSUM_FW_PATH ="/mmlabstorage/workingspace/VideoSum/videosummarizationframework/"
sys.path.append(os.path.join(VIDEOSUM_FW_PATH,'source/config')) #config path append
sys.path.append(os.path.join(VIDEOSUM_FW_PATH,'source/utilities'))
from config import cfg
from check_permission import check_permission_to_write
from parse_csv import get_metadata

#make log config
log_name = os.path.join(VIDEOSUM_FW_PATH,'source/log','feature-extract-log.txt')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=log_name,
                    filemode='a')
"""[class extract feature]
Input: dataset name, output path
Output: được quyền lựa chọn kiểu file output, path out
+.npy
+.h5
used: ExtractFeature(input_argv).VGG16(tầng,out_extention)

bao gồm các bước:
    +đọc dữ liệu
        Nếu nó là video:
        Nếu nó là frames:
    +trích xuất feature
    +tổng hợp feature
    +ghi ra file
kiểm tra path trước khi lưu
""" 

class ExtractFeature:
    def __init__(self,dataset_name,output_path,sampling_rate=1,extension_out='npy',f_or_v='video',device_name='GPU:0'): 
        """[INIT fuction]
        Arguments:
            dataset_name {string} -- [Name of dataset]
            output_path {string} -- [Directory for output results]

        Keyword Arguments:
            sampling_rate {int} -- [sampling rate] (default: {'1'})
            extension_out {str} -- [file extension of output result (npy or h5) ] (default: {'npy'})
            f_or_v {str} -- [input is video or foler of frame video] (default: {'video'})
            device_name {str} -- [device] (default: {'GPU:0'})
        Author: thinhplg
        """
        self.dataset = dataset_name
        self.output = output_path
        self.frames_or_video = f_or_v
        self.device = '/device:'+device_name
        self.extension = extension_out
        self.samplingRate = sampling_rate
        if check_permission_to_write(self.output) is False:
            sys.exit()


    def VGG19(self,output_layer='fc2'):
        """VGG1() - fc2 - output_shape = 4096
        """
        base_model = vgg19.VGG19(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = vgg19.preprocess_input
        return self._process()

    def VGG16(self,output_layer='fc2'):
        """VGG16 - fc2 - output_shape = 4096
        """
        base_model = vgg16.VGG16(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = vgg16.preprocess_input
        return self._process()

    def ResNet50(self,output_layer='avg_pool'):
        """ResNet50 - avg_pool - output_shape = 2048
        """
        base_model = resnet.ResNet50(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = resnet.preprocess_input
        return self._process()

    def ResNet152(self,output_layer='avg_pool'):
        """ResNet152 - avg_pool - output_shape = 2048
        """
        base_model = resnet.ResNet152(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (224,224)
        self.model = _model
        self.preinput = resnet.preprocess_input
        return self._process()

    def InceptionV3(self,output_layer='avg_pool'):
        """ResNet50 - avg_pool - output_shape = 2048
        """
        base_model = inception_v3.InceptionV3(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (299,299)
        self.model = _model
        self.preinput = inception_v3.preprocess_input
        return self._process()
        pass

    def _process(self):
        """Fuction for run process
        Check input is video folder or frame folder
        """
        fov = self.frames_or_video 
        if fov == 'v' or fov =='video' or fov =='videos':
            try:
                with tf.device(self.device):
                    self.__process_video()
            except Exception as e:
                logging.error(e)
        elif fov == 'f' or fox =='frame' or fov =='frames':
            self.__process_frame()
        else:
            pass

    def __process_frame(self):
        pass

    def __process_video(self):
        videoName,videoPathLists,_,nFrame,_ = self._read_meta_data()
        for idx,video in enumerate(videoName):
            feat = []
            print("%s/%s : %s with sampling rate is %d"%(idx+1,len(videoName),video,self.samplingRate))
            path = videoPathLists[idx]
            print(path)
            vidcap = cv2.VideoCapture(path)
            if (vidcap.isOpened()== False):
                #check opened?
                logging.error("Cant open video: %s path: "%(video,path))
                continue
            sam = self.samplingRate
            pbar = tqdm(total = nFrame)
            it = 0
            while(vidcap.isOpened()):
                pbar.update(1)
                suc, img = vidcap.read()
                it+=1
                if(suc == False):
                    break
                if ((it-1)%sam) != 0:
                    continue
                _feature = self.__extract(img)
                feat.append(_feature)
            result = np.asarray(feat)
            result = np.squeeze(result)
            namefile = os.path.splitext(os.path.basename(video))[0]
            self._write_to_file(namefile,result)

    def _read_meta_data(self):
        #read data from file csv
        dn = self.dataset   #dataset name
        if dn == 'bbc' or dn =='BBC' or dn == 'BBC EastEnders':
            namevid,path,fps,nFrame,duration = get_metadata('bbc')
        elif dn == 'summe' or dn =='SumMe' or dn == 'SUMME':
            namevid,path,fps,nFrame,duration = get_metadata('summe')
        else:
            namevid,path,fps,nFrame,duration = get_metadata(dn)

        return namevid,path,fps,nFrame,duration

    def _write_to_file(self,name,data):
        #Write feature data to file
        if self.extension == 'npy':
            self.__write_to_file_npy(name,data)
        else:
            self.__write_to_file_h5(name,data)

    def __write_to_file_npy(self,name,data):
        path = os.path.join(self.output,
                            name+'_'+str(self.samplingRate)+'.npy')
        np.save(path,data)
        logging.info("Done write file %s.npy at %s"%(name,path))

    def __write_to_file_h5(self):
        pass

    def __extract(self,input_image):
        #Extract feature from image                    
        img = Image.fromarray(input_image)
        img = img.resize(self.imageSize) #tùy thuộc vào phương pháp thì cái reesize này khác nao
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = self.preinput(img_data)
        _feature = self.model.predict(img_data)
        return _feature

def main():
    #Example for runing
    ex = ExtractFeature('tvsum','./',sampling_rate=5).VGG16()

if __name__ == '__main__':
    main()
    