#keras import
from keras.preprocessing import image
from keras.applications import vgg16,vgg19,resnet,inception_v3,resnet_v2
from keras.models import Model
from keras.layers import Layer

#Some code here. No need to modify it.

class ExtractFeatureVideo(Feature):
    def __init__():  
        #Some code here. No need to modify it.
        #........
        #........

    def NAME_OF_CNN_HERE(self,output_layer=OUTPUT_LAYER_HERE):

        self._method = 'name_of_cnn_here'
        self._framework = 'tensorflow'
        self._device = '/device:'+self._device 
        base_model = tensorflow_model_here.TENSORFLOW_MODEL_HERE(weights='imagenet')
        _model = Model(input=base_model.input, output=base_model.get_layer(output_layer).output)
        _model.summary()
        self.imageSize = (XXX_HERE,YYY_HERE) #size of image
        self.model = _model
        self.preinput = tensorflow_model.preprocess_input 
        return self._process()           
        """In this function must change. Example
        NAME_OF_CNN_HERE: GoogleNet, ResNet, or whatever you want.
        tensorflow_model_here: resnet50, inceptionv3. Check this link for more information https://keras.io/api/applications/
        TENSORFLOW_MODEL_HERE: Name of model. Check link above
        XXX_HERE and YYY_HERE: Size of input model need
        """


        #Some code here. No need to modify it.
        #........
        #........
#This class use to extract feature for all video from a dataset. Inheritance class ExtractFeatureVideo
class ExtractFeatureDataSet(ExtractFeatureVideo):

#END OF FILE