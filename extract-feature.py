#This code support 3 dataset: Summe, Tvsum50, BBC. To change data set. Try to change 2 variable: data_path, output_path
#To run this script use command "python vgg16_fe.py xyz" with xyz is the name of dataset.

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Layer
import numpy as np
import cv2 
from PIL import Image
import multiprocessing
import sys

import os
from tqdm import tqdm
#data_path = '/mmlabstorage/workingspace/VideoSum/videosummarizationframework/data/TVSum_processed_data/tvsum50_frames'
#BBC data
#Path for input data and output
path_out = '/mmlabstorage/workingspace/VideoSum/videosummarizationframework/data'
bbc =['/mmlabstorage/datasets/TRECVID/TRECVID_BBC_EastEnders/videos',path_out+'/BBC_processed_data/time_shots_bbc/feature_vgg']
tvsum = ['/mmlabstorage/datasets/TVSum50/ydata-tvsum50-v1_1/video',path_out+'/TVSum_processed_data/time_shots_tvsum50/feature_vgg']
summe = ['/mmlabstorage/datasets/SumMe/videos',path_out+'/SumMe_processed_data/time_shots_summe/feature_vgg']

dic = {'bbc':bbc,'tvsum':tvsum,'summe':summe}

#input name of dataset
_input = sys.argv[1]

#set direction for data and output
data_path = dic[_input][0]
output_path = dic[_input][1]
print(data_path,output_path)




#Function extract feature
def extract_feature(data_path,output_path):

    #load model
    base_model = VGG16(weights='imagenet')#, include_top=True)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    model.summary()

    included_extenstions=['mp4']

    #load list of video name
    videoLists=[fn for fn in os.listdir(data_path) if any([fn.endswith(ext) for ext in included_extenstions])]

    #load finished video to remove from run list
    donefile = [os.path.splitext(os.path.basename(f))[0]+'.mp4' for f in os.listdir(output_path)] 

    #run list
    videoList = [item for item in videoLists if item not in donefile]

    for idx,video in enumerate(videoList):
        feat=[]
        print("%s/%s : %s"%(idx+1,len(videoList),video))
        vidcap = cv2.VideoCapture(os.path.join(data_path,video))
        if (vidcap.isOpened()== False):
            #check opened?
            with open(output_path+'/Failed.txt','w') as file:
                file.write(video+'/n')
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
        #save to file
        np.save(output_path+'/'+namefile+'.npy',result)



def main():
    extract_feature(data_path,output_path)


if __name__ == '__main__':
    main()
