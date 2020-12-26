# **Traffic Sign Recognition**

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the neural network's state with test images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

[data_histogram_for_class]: ./image/data_histogram_for_class.png "Data Histogram for Class"

[balancing_data]: ./image/balancing_data.png "Balancing Data"

[model_accuracy]: ./image/model_accuracy.png "Model Accuracy"

[preprocessing_0]: ./image/preprocessing_0.png "Proprocessing 0"
[preprocessing_1]: ./image/preprocessing_1.png "Proprocessing 1"
[preprocessing_2]: ./image/preprocessing_2.png "Proprocessing 2"

[image_augmentation]: ./image/image_augmentation.png "Image Augmentation"

[learning_rate]: ./image/learning_rate.png "Learning Rate"

[other_image_01]: ./external_image/01.jpg "Other Traffic Sign 1"
[other_image_02]: ./external_image/02.jpg "Other Traffic Sign 2"
[other_image_03]: ./external_image/03.jpg "Other Traffic Sign 3"
[other_image_04]: ./external_image/04.jpg "Other Traffic Sign 4"
[other_image_05]: ./external_image/05.jpg "Other Traffic Sign 5"
[other_image_06]: ./external_image/06.jpg "Other Traffic Sign 6"
[other_image_07]: ./external_image/07.jpg "Other Traffic Sign 7"
[other_image_08]: ./external_image/08.jpg "Other Traffic Sign 8"
[other_image_09]: ./external_image/09.jpg "Other Traffic Sign 9"
[other_image_10]: ./external_image/10.jpg "Other Traffic Sign 10"

[top5_01]: ./image/top5_01.png "Top 5 for Sign 1"
[top5_02]: ./image/top5_02.png "Top 5 for Sign 2"
[top5_03]: ./image/top5_03.png "Top 5 for Sign 3"
[top5_04]: ./image/top5_04.png "Top 5 for Sign 4"
[top5_05]: ./image/top5_05.png "Top 5 for Sign 5"
[top5_06]: ./image/top5_06.png "Top 5 for Sign 6"
[top5_07]: ./image/top5_07.png "Top 5 for Sign 7"
[top5_08]: ./image/top5_08.png "Top 5 for Sign 8"
[top5_09]: ./image/top5_09.png "Top 5 for Sign 9"
[top5_10]: ./image/top5_10.png "Top 5 for Sign 10"

[visualization_layer_1]: ./image/visualization_layer_1.png "Layer 1 Visualization"
[visualization_layer_2]: ./image/visualization_layer_2.png "Layer 2 Visualization"
[visualization_layer_3]: ./image/visualization_layer_3.png "Layer 3 Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## Data Set Summary & Exploration

### 1. Basic summary of the data set.

The numpy library is used to calculate summary statistics of the traffic signs data set:

|Item                          |Description	| 
|:-----------------------------|:----------:| 
|Number of training examples   |34799       | 
|Number of validation examples |4410        | 
|Number of testing examples    |12630       | 
|Image data shape              |(32, 32, 3) | 
|Range of image data           |0 - 255     | 
|Number of classes             |43          | 

The following show the different classes of the German traffic signs.

|Class index|Class name                                        | 
|:---------:|:-------------------------------------------------| 
|1          |Speed limit (20km/h)                              |
|2          |Speed limit (30km/h)                              |
|3          |Speed limit (50km/h)                              |
|4          |Speed limit (60km/h)                              |
|5          |Speed limit (70km/h)                              |
|6          |Speed limit (80km/h)                              |
|7          |End of speed limit (80km/h)                       |
|8          |Speed limit (100km/h)                             |
|9          |Speed limit (120km/h)                             |
|10         |No passing                                        |
|11         |No passing for vehicles over 3.5 metric tons      |
|12         |Right-of-way at the next intersection             |
|13         |Priority road                                     |
|14         |Yield                                             |
|15         |Stop                                              |
|16         |No vehicles                                       |
|17         |Vehicles over 3.5 metric tons prohibited          |
|18         |No entry                                          |
|19         |General caution                                   |
|20         |Dangerous curve to the left                       |
|21         |Dangerous curve to the right                      |
|22         |Double curve                                      |
|23         |Bumpy road                                        |
|24         |Slippery road                                     |
|25         |Road narrows on the right                         |
|26         |Road work                                         |
|27         |Traffic signals                                   |
|28         |Pedestrians                                       |
|29         |Children crossing                                 |
|30         |Bicycles crossing                                 |
|31         |Beware of ice/snow                                |
|32         |Wild animals crossing                             |
|33         |End of all speed and passing limits               |
|34         |Turn right ahead                                  |
|35         |Turn left ahead                                   |
|36         |Ahead only                                        |
|37         |Go straight or right                              |
|38         |Go straight or left                               |
|39         |Keep right                                        |
|40         |Keep left                                         |
|41         |Roundabout mandatory                              |
|42         |End of no passing                                 |
|43         |End of no passing by vehicles over 3.5 metric tons|

### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

It is a bar chart showing how different classese are disbributed in the training data, the validation data, and the testing data respectively.

![data_histogram_for_class]

## Design and Test a Model Architecture

### 1. Image Data Preprocssing


#### 1. Balancing data, converting to grayscale and normalization


The precessing is dividied into 3 step.
1. Balancing data.

The training set is not well balanced for different classes. Therefore, duplicate the training samples for each class randomly so all classes have the same numbers of samples. Different sets of duplicates are computed for each epoch of the training. After the copying, many classes would have the same sample. To overcome this problem, we would perform image augmentation for each set later.
![balancing_data]

2. Converting to grayscale.

Convert to grayscale and make the data between 0 and 1. They are used for initial data.
||
|:------:|
|Original|
|![preprocessing_0]|
|Grayscale|
|![preprocessing_1]|

3. Normalization.

Normalize the result so that the 0 is mapped to the minimum value and 255 is mapped to the maximum value for each image. They are used right before the training or evaluation.
||
|:------:|
|Normalizing|
|![preprocessing_2]|

#### 2. Image augmentation

Additional data would be generated because the following reasons.
1. The training set is not well balanced for different classes. Image augmentation is necessary to make sure that there are no the same.
2. It can decrease the change of over-fitting.
3. More data can be used to train the model.

Image Augmentation is done using different ways:
1. Random lighting by adjusting gamma
2. Random rotation
3. Random scale
4. Random translation

The following figure illustrates the same image applying all the previous operations randomly.
![image_augmentation]

#### 2. Model architecture.

My model consists of the following layers:

|Layer             |Description                 |Output       |Parameter|
|:-----------------|:---------------------------|:-----------:|:-------:|
|Input             |Grayscale image             |32 x 32 x 1  |0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |30 x 30 x 64 |576      |
|RELU              |Relu Activation             |30 x 30 x 64 |0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |28 x 28 x 64 |36864    |
|RELU              |Relu Activation             |28 x 28 x 64 |0        |
|Max pooling 2 x 2 |2 x 2 stride                |14 x 14 x 64 |0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |12 x 12 x 128|73728    |
|RELU              |Relu Activation             |12 x 12 x 128|0        |
|Max pooling 2 x 2 |2 x 2 stride                |6 x 6 x 128  |0        |
|Flatten           |Flatten                     |4608         |0        |
|Dense             |Dense network               |1024         |4719616  |
|RELU              |Relu Activation             |1024         |0        |
|Drop out          |80 % Passing                |1024         |0        |
|Dense             |Dense network               |512          |524800   |
|RELU              |Relu Activation             |512          |0        |
|Drop out          |80 % Passing                |512          |0        |
|Dense             |Dense network               |43           |22059    |
|Softmax           |Softmax                     |43           |0        |
 
#### 3. Training the model

The model is trained using Adam Optimizer with the following parameters.
|Item            |Value|
|:---------------|:---:|
|Batch size      |512  |
|Number of epochs|15   |
|Learning rate   |0.001|

#### 4. The approach for finding hte solution and getting high accuracy of the validation set.

The final model results are:

|Item                       |Value   |
|:--------------------------|:------:|
|Accuracy of training set   |99.759 %|
|Accuracy of validation set |98.413 %|
|Accuracy of test set       |97.348 %|

All of them are over 97 %.

The following figure show the accuracy against the epochs.

![model_accuracy]


The first architecture that was tried is LeNet as its function is very similar. LeNet structure is quite simple and used to predict 10 different digits from grayscale image while our application is to predict 43 different traffic signs from color images. LeNet consists of consists of some convolution layers hiwch work well for image becuase of the weight sharing for different part of the image.

However, when the LeNet is used without balancing the data set for different classes and image augmentation. it is discovered the accuracy of the training set is only around 90 % and the accuracy of the validation is even lower. Therefore, I try to increase the depth by one more convolution layer, but decrease the size of the kernel. At the same time, I also make the network wider to increase the capacity of the network so it can increase the acccuracy of the training set.

To overcome the over-fitting and minimize the gap between the accuracies of the training set and the validation set, image augmentation is performed. At the same time, data is generated to balancing the data set for different classes. In addition, dropout layer is used to prevent the deeper and wider network become over-fitting.

The learning rate is chosen to be 0.001. I have tried different learning rate 0.0001, 0.001, 0.01, and 0.1. Learning rate larger than 0.001 cannot achieve high accuracy while learning rate smaller than 0.001 made the learning too slow.

![learning_rate]

The epoch is set to 15 as I found that the accuracies of the training set and the validation set cannot increase after around 10 - 15.

### Test the model on new images

#### 1. Test the model on 10 pictures of German traffic signs either found from the web or taken by myself.

The following table shows the 10 German traffic signs tested:

|File Name|Image            |Label                    |
|:--------|:---------------:|:------------------------|
|01.jpg   |![other_image_01]|Road narrows on the right|
|02.jpg   |![other_image_02]|Speed limit (50km/h)     |
|03.jpg   |![other_image_03]|No entry                 |
|04.jpg   |![other_image_04]|Road work                |
|05.jpg   |![other_image_05]|Go straight or right     |
|06.jpg   |![other_image_06]|Keep right               |
|07.jpg   |![other_image_07]|Yield                    |
|08.jpg   |![other_image_08]|Double curve             |
|09.jpg   |![other_image_09]|No entry                 |
|10.jpg   |![other_image_10]|No vehicles              |



#### 2. Discussion of the performance of the model on new images.

Here are the results of the prediction:

|File Name|Image            |Label                    |Prediction               |
|:--------|:---------------:|:------------------------|:------------------------|
|01.jpg   |![other_image_01]|Road narrows on the right|Road narrows on the right|
|02.jpg   |![other_image_02]|Speed limit (50km/h)     |Speed limit (50km/h)     |
|03.jpg   |![other_image_03]|No entry                 |No entry                 |
|04.jpg   |![other_image_04]|Road work                |Road work                |
|05.jpg   |![other_image_05]|Go straight or right     |Go straight or right     |
|06.jpg   |![other_image_06]|Keep right               |Keep right               |
|07.jpg   |![other_image_07]|Yield                    |Yield                    |
|08.jpg   |![other_image_08]|Double curve             |Double curve             |
|09.jpg   |![other_image_09]|No entry                 |No entry                 |
|10.jpg   |![other_image_10]|No vehicles              |No vehicles              |

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100 %. This is similar to the accuracies of the validation set and the test set.

#### 3. Top 5 softmax probabilities for each image .

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Top 5 Probablities for 01.jpg

Road narrows on the right (Class 24)
					96.552%
General caution (Class 18)
					3.448%
Pedestrians (Class 27)
					0.000%
Double curve (Class 21)
					0.000%
Children crossing (Class 28)
					0.000%

![top5_01]

Top 5 Probablities for 02.jpg

|Class|Probability|
|:----|:---------:|
|Speed limit (50km/h) (Class 2)|99.992 %|
|Speed limit (60km/h) (Class 3)|0.008 %|
|Speed limit (30km/h) (Class 1)|0.000 %|
|Speed limit (80km/h) (Class 5)|0.000 %|
|No passing for vehicles over 3.5 metric tons (Class 10)|0.000 %|

![top5_02]

Top 5 Probablities for 03.jpg

|Class|Probability|
|:----|:---------:|
|No entry (Class 17)|99.912 %|
|Stop (Class 14)|0.085 %|
|Keep right (Class 38)|0.002 %|
|Priority road (Class 12)|0.000 %|
|No passing (Class 9)|0.000 %|

![top5_03]

Top 5 Probablities for 04.jpg

|Class|Probability|
|:----|:---------:|
|Road work (Class 25)|99.620 %|
|Children crossing (Class 28)|0.379 %|
|Bicycles crossing (Class 29)|0.001 %|
|Beware of ice/snow (Class 30)|0.000 %|
|Road narrows on the right (Class 24)|0.000 %|

![top5_04]

Top 5 Probablities for 05.jpg

|Class|Probability|
|:----|:---------:|
|Go straight or right (Class 36)|99.992 %|
|Ahead only (Class 35)|0.006 %|
|Turn right ahead (Class 33)|0.002 %|
|No passing for vehicles over 3.5 metric tons (Class 10)|0.000 %|
|Stop (Class 14)|0.000 %|

![top5_05]

Top 5 Probablities for 06.jpg

|Class|Probability|
|:----|:---------:|
|Keep right (Class 38)|100.000 %|
|No passing for vehicles over 3.5 metric tons (Class 10)|0.000 %|
|Turn left ahead (Class 34)|0.000 %|
|Priority road (Class 12)|0.000 %|
|Stop (Class 14)|0.000 %|

![top5_06]

Top 5 Probablities for 07.jpg

|Class|Probability|
|:----|:---------:|
|Yield (Class 13)|100.000 %|
|Priority road (Class 12)|0.000 %|
|Stop (Class 14)|0.000 %|
|Speed limit (30km/h) (Class 1)|0.000 %|
|No vehicles (Class 15)|0.000 %|

![top5_07]

Top 5 Probablities for 08.jpg

|Class|Probability|
|:----|:---------:|
|Double curve (Class 21)|82.911 %|
|Children crossing (Class 28)|16.958 %|
|Beware of ice/snow (Class 30)|0.116 %|
|Right-of-way at the next intersection (Class 11)|0.013 %|
|Wild animals crossing (Class 31)|0.001 %|

![top5_08]

Top 5 Probablities for 09.jpg

|Class|Probability|
|:----|:---------:|
|No entry (Class 17)|99.957 %|
|Stop (Class 14)|0.043 %|
|Turn right ahead (Class 33)|0.000 %|
|Roundabout mandatory (Class 40)|0.000 %|
|Yield (Class 13)|0.000 %|

![top5_09]

Top 5 Probablities for 10.JPG

|Class|Probability|
|:----|:---------:|
|No vehicles (Class 15)|100.000 %|
|No passing (Class 9)|0.000 %|
|Speed limit (70km/h) (Class 4)|0.000 %|
|Keep right (Class 38)|0.000 %|
|Speed limit (50km/h) (Class 2)|0.000 %|

![top5_10]

#### 4. Visualizing the Neural Network


![visualization_layer_1]
![visualization_layer_2]
![visualization_layer_3]
