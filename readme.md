# extract-feaute.py (Convolutional Neural Network Feature Extractor)
__Version__: 0.3
## Requirements
The main library
``` bash
tensorflow
tensorflow-gpu
keras
torch
torchvision
opencv-python
and more . . . 
````
___
## Usage on another script
Two class ``ExtractFeatureVideo`` and ``ExtractFeatureDataSet``

Import to use

``` bash
from CNN import ExtractFeatureVideo, ExtractFeatureDataSet
```
### Extract Feature Video

`` module = ExtractFeatureVideo(video_path,sampling_rate=int,device_name='str')
``

`` feature = module.VGG16(output_layer='str')
``

`` feature.save(path_to_save)
``

This code support VGG16, VGG19, ResNet50, ResNet152, InceptionV1, InceptionV3. To use anoher architecture read tutorial below.

--

### Extract Feature DataSet

This class is inheritance of ExtractFeatureVideo. 

`` module = ExtractFeatureDataSet(dataset_name,output_path,from_id=int,to_id=int,sampling_rate=int,device_name='str'):
``

``module.VGG19(output_layer='str')
``

---

## Usage CNN.py script

Use command ``python CNN.py --help`` or ``python CNN.py -h``  for see full arguments

### Positional arguments

``python CNN.py name_of_method x_str
``

``name_of_method`` is name of CNN architecture. Support: VGG16, VGG19, ResNet50, ResNet152, InceptionV1, InceptionV3

``x_str`` is Video(V) or Dataset(D). Choose Video or DataSet for run

#### Example

``python CNN.py inceptionv1 v``  |    ``python CNN.py inceptionv1 d``

``python CNN.py Resnet50 video``  |    ``python CNN.py resnet50 dataset`` 

### Optional arguments

Include positional arguments, and have two optional ``+Video`` and ``+Dataset``

``+Video``

+ __*VideoPath**__ : Path to video

+ __*-sr or --samplingRate*__: Sampling rate, default = 1.

+ __*-l or --layer*__: Name of hiden layer on CNN architecture, default=None.

+ __*-d or --device*__: Device use to run, default='0'.

+ __*-s or --save*__: Path to save feature data to file, default=None.
    + Example: ``python CNN.py inceptionv1 video +Video /path/to/video.mp4 -sr 2 -d 1 -s path/to/folder/save/ -l fc1``

``+DataSet``

+ __*DataSetName**__ : Name of Dataset. Support three dataset: _*BBC EastEnders, TVSum50, SumMe*_. To use on another dataset, read tutorial below.

+ _OutputPath*_ : Path to folder save feature data file. 

+ __*-f or --fromid*__ : ID begin-th video, default=None.

+ __*-e or --endid*__ : ID end-th video. Two args use for big dataset. Run from video begin-th to video end-th, default=None.

+ __*-sr or --samplingRate*__: Sampling rate, default = 1.

+ __*-d or --device*__: Device use to run, default='0'.
    
    + Example: ``python CNN.py resnet152 dataset +DataSet bbc /path/folder/out -f 12 -e 35 -sr 2 -d 2``

_*To use on another dataset. Try to use file utilities/make_csv_from_dataset.py with command*_

``python make_csv_from_dataset.py +newdataset dataset_name /path/to/folder/video /path/to/folder/out``

---

## Add new CNN architector

There are two famous frameworks used for extract feature is ``tensorflow`` and ``pytorch``. I make two template for them.

### Tensorflow 

[Read this template code.](https://github.com/PhamThinh31/src-for-videosummarization/blob/master/template-tensorflow.py)

Brief code:

```
def NAME_OF_CNN_HERE(self,output_layer=OUTPUT_LAYER_HERE):
    ...
    base_model = tensorflow_model_here.TENSORFLOW_MODEL_HERE(weights='imagenet')
    ...
    self.imageSize = (XXX_HERE,YYY_HERE) #size of image
    ... 
```
Change the content of the lines that have the words *HERE. Read comment on template code

### Pytorch 

[Read this template code.](https://github.com/PhamThinh31/src-for-videosummarization/blob/master/template-pytorch.py)

Brief code:

```
class NAME_OF_CNN_HERE(nn.Module):
        ---
        change_pytorch_model_here = models.CHANGE_PYTORCH_MODE_HERE(pretrained=True)
        ...
        change_pytorch_model_here.float()
        change_pytorch_model_here.cuda()
        change_pytorch_model_here.eval()
        module_list = list(CHANGE_PYTORCH_MODEL_HERE.children())
        self.conv5 = nn.Sequential(*module_list[: -XXX_HERE]) 
    ---
```
Change the content of the lines that have the words *HERE. Read comment on template code
