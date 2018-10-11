Keras InceptionV3 모델 시각화
===========================
Visualize filters and saliencies of convolutional layer in Keras InceptionV3 model. <br/>
Using lucid, visualize an image that activates target filter. (feature visualization) <br />
Using Keras-vis, visualize an image which shows where that filter looks in input image (saliency). <br /> 

Dependencies
------------
- tensorflow
- keras
- keras-vis
- lucid
- matplotlib
- numpy

How to Use
------
example
```
$python visualize.py
-modelzoo_file="keras_inception_v3_frozen.pb.modelzoo"
-conv_layer_idx=6
-filter_idx=15
-num_of_filters=1
-input_image_path="image/welshcorgi.jpg"
```

Output
------
feature_visualization output (conv2d_6 15th filter) <br/>
![conv2d_6 15th filter](Conv6_15.png)  <br/>


saliency_visualization output (conv2d_6 15th filter saliency)  <br/>
![conv2d_6 15th filter saliency](Conv6_15_saliency.png)  <br/>

option 별 설명
--------------
-modelzoo_file:
A .modelzoo file which is outcome of freeze_graph.py
([lucid model import](https://colab.research.google.com/drive/1PPzeZi5sBN2YRlBmKsdvZPbfYtZI-pHl)

-conv_layer_idx:
Index of convolutional layer in Keras Inceptionv3 which you want to visualize. Range 1~94

-filter_idx:
Index of filter in the convolutional layer which you want to visualize. (Optional)

-num_of_filters:
The number of filters you want to visualize at the layer. The 'num_of_fileters' of filters that most activates the layer will visualized. If you want to visualize specific filter, then this should be 1. 

-input_image_path:
A Path of image you want to visualize the saliency.
