#This code support 3 dataset: Summe, Tvsum50, BBC. To change data set. Try to change 2 variable: data_path, output_path
#To run this script use command "python vgg16_fe.py xyz" with xyz is the name of dataset.

from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Layer
import numpy as np
import cv2 
from PIL import Image
from keras.utils import multi_gpu_model
from checkper import check_permission_to_write

import multiprocessing
import sys
import tensorflow as tf
import os
from tqdm import tqdm
#data_path = '/mmlabstorage/workingspace/VideoSum/videosummarizationframework/data/TVSum_processed_data/tvsum50_frames'
#BBC data
#Path for input data and output

path_out = '/mmlabstorage/workingspace/VideoSum/videosummarizationframework/data'
bbc =['/mmlabstorage/datasets/TRECVID/TRECVID_BBC_EastEnders/videos','/home/sum/thinhplg/BBC']#path_out+'/BBC_processed_data/time_shots_bbc/feature/VGG19']
tvsum = ['/mmlabstorage/datasets/TVSum50/ydata-tvsum50-v1_1/video','/home/sum/thinhplg/tvsum']#path_out+'/TVSum_processed_data/time_shots_tvsum50/feature_vgg']
summe = ['/mmlabstorage/datasets/SumMe/videos',path_out+'/SumMe_processed_data/time_shots_summe/feature_vgg']

dic = {'bbc':bbc,'tvsum':tvsum,'summe':summe}

#input name of dataset
_input = sys.argv[1]

x = int(sys.argv[2])
y = int(sys.argv[3])
namedevice = (sys.argv[4])

#set direction for data and output
data_path = dic[_input][0]
output_path = dic[_input][1]
if(_input=='bbc'):
	outdir = '/mmlabstorage/workingspace/VideoSum/videosummarizationframework/data/BBC_processed_data/time_shots_bbc/feature/VGG19'
	if(check_permission_to_write(outdir)) is True:
		output_path = outdir
print(data_path,output_path)




#Function extract feature
def extract_feature(data_path,output_path,from_x,to_y,):

    #load model
    base_model = VGG19(weights='imagenet')#, include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    model.summary()

    included_extenstions=['mp4']

    #load list of video name
    videoLists=[fn for fn in os.listdir(data_path) if any([fn.endswith(ext) for ext in included_extenstions])]

    #temp variable 
    videoLists = videoLists[from_x:to_y]
    #load finished video to remove from run list
    donefile=[]
    with open(output_path+'/done.txt','r') as file:
        temp = file.readlines()
        donefile = [os.path.splitext(os.path.basename(f.rstrip()))[0]+'.mp4' for f in temp]
    print("Had done %d file"%len(donefile)) 

    #run list
    videoList = [item for item in videoLists if item not in donefile]

    for idx,video in enumerate(videoList):
        feat=[]
        print("%s/%s : %s"%(idx+1,len(videoList),video))
        vidcap = cv2.VideoCapture(os.path.join(data_path,video))
        if (vidcap.isOpened()== False):
            #check opened?
            with open(output_path+'/failed.txt','a') as file:
                file.write(video+'\n')
                file.close()
            continue
        #total frame of this video
        nFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        pbar = tqdm(total = nFrames)
        while(vidcap.isOpened()):
            pbar.update(1)
            suc, img = vidcap.read()
            if(suc == False):
                break
            img = Image.fromarray(img)
            img = img.resize((224,224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            vgg16_feature = model.predict(img_data)
            feat.append(vgg16_feature)
        result = np.asarray(feat)
        result = np.squeeze(result)
        namefile = os.path.splitext(os.path.basename(video))[0]
        with open(output_path+'/done.txt','a') as file:
            file.write(video+'\n')
            file.close()
        #save to file
        np.save(output_path+'/'+namefile+'.npy',result)



def main():
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    try:
        with tf.device('/device:'+namedevice):
            extract_feature(data_path,output_path,x,y)
    except RuntimeError as e:
        with open(output_path+'/failed.txt','a') as file:
            file.write(str(e)+'\n')



if __name__ == '__main__':
    main()