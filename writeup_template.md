#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/initial_hist.jpg "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_30_48_287.jpg "Grayscaling"
[image3]: ./examples/left_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image4]: ./examples/right_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image5]: ./examples/after_hist.png "Recovery Image"
[image6]: ./examples/original.jpg "Normal Image"
[image7]: ./examples/sheared.jpg "ShearedImage"
[image8]: ./examples/cropped.jpg "Cropped Image"
[image9]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.py video of the model driving in autonomous mode
* examples folder containg some samples from the processing layer

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model chosen is a convolutional neural network, which was adapted from the comma.ai model. It uses 4 Convolutional layers followed by 2 fully connected layers feeding into a final neuron which outputs the steering angle. An initial lambda layer normalizes the data. The input data for the model is 64x64x3 images. The nonlinearity activation used is the ELU, while the loss function is mean squared error.

####2. Attempts to reduce overfitting in the model

The two main attempts done at reducing overfitting is generating/collecting lots of data besides the udacity provided samples. The data generated captures different situtations that helps generalize the model.

Also, in the model itself I added two dropout layers before each fully connected layer - I use hyperparameters 0.2 and 0.5 for dropout. The model is created in the method comma_ai_model(). 

####3. Model parameter tuning

I decided to use an adam optimizer, so the learning rate was not tuned manually. One less hyperparameter to worry about.

####4. Appropriate training data

Since i had issues getting very good training data - even with a mouse, i can't say i'm really glad with my driving :) - i tried an approach of generating a lot of artificial data from what training data i already had, and from the udacity samples. 


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to progress from gettinig the car to move, all the way to actulaly drive around the track. I chose to implement a variant of the comma.ai model (comm_ai_model method in model.py) which has been used succesfully for this type of problem.

Running it only on provided samples proves that data is everything. The model was initially only capable of driving forward and small steering because of the straing line bias and overfitting. Clearly a lot more data was needed. Also dropout layers were used in order to reduce overfitting. 

A good gauge of how well the model was working was actually trying it on track 1. This proved that low mse doesn't necessarily mean a good driving behavior.

At the end of the process of gathering the data, augmenting it, and tuning hyperparameters, the vehicle is able to drive autonomously around the track without leaving the road.


####2. Final Model Architecture

The final model architecture is a variation of the comma.ai model(model.py comma_ai_model method). It consists of a convolution neural network with the following layers and layer sizez:

- Lambda normalizing layer - data between 0 and 1 with a mean of 0
- Convolutional layer 32 depth 8x8 kernel 4x4 stride
- Convolutional layer 64 depth 8x8 kernel 2x2 stride
- Convolutional layer 128 depth 4x4 kernel 1x1 stride
- Convolutional layer 128 depth 2x2 kernel 1x1 stride
- Dropout(0.2)
- fully connected layer - 128 neurons
- Dropout(0.5)
- fully connected layer - 128 neurons

The activation function used is the ELU.

####3. Creation of the Training Set & Training Process

The udacity provided samples set is pretty good, but not enough by itself to traing a working model. That means i had to do some extra recording of data from the simulator. I drove some laps and got some data, but the model still had trouble at certain curbs. I tried to recorder some recovery laps but i couldn't seem to master the simulator well enought. Reading through the nanodegree slack channel i learned that a perfectly good approach might be do use the initial data i had and try to augment the dataset with more images. After much fiddling i settled on the follwing processing pipeline.

The first stage is randomly reading the data. Since we have quite a lot of data we don't want to keep it in memory all the time. A better approach is to use a python generator which gets the data in batches and feeds it to the model. I read data from all three cameras - chose left/center/right camera at random. The output steering angle is adjusted with +/- 0.27 for the left and right cameras. Here is an example of left/center/right camera input data:

![alt text][image3]

![alt text][image2]

![alt text][image4]

The second stage is a random warp of the image. Since a lot of the track is strainght driving we can see in this histogram of the initial data that there is an overwhelming number of small steering angle samples. We need a to equalize the histogram a bit in order to prevent the bias from transferring inside the model. This means generating images which correspond to a higher absolute steering angle. An approach discussed in the slack channel(Thanks Vivek and Kaspar) is to take the bottom half of the image and warp the perspective left or right randomly using a uniform distribution while adjusting the angle to match the new image. After running the generator a few times heres an updated histogram. It looks like the percentage of small angles is decreasing. 

![alt text][image1]

![alt text][image5]


Here is an example of input and output from this stage:

![alt text][image6]

![alt text][image7]

The third stage of the pipeline is croping the bottom and top part of the image. We don't really need these parts, since the bottom is mostly a view of the front of the car, while the top is mostly sky and some scenery. Also we take this opportunity and resize the image to make training a lot faster. Here is an output from this stage:

![alt text][image8]

The next stage is a decision to randomly flip the image horizontally while adjusting the angle. This should avoid a bias from track 1 with left curves. Here is an output of this stage:

![alt text][image9]

The final stage is adjusting the brightness in the image. This should create samples with different ammounts of brightness which should help generalize the model. 

The validation set is mostly the initial set cropped and resized. I decided to do this since the training set is heavily augmented by the processing pipeline. The inital samples should provide a sufficient test baseline.

The training was done for 10 epochs of 100 batches of 256 samples each.

Conclusion:
Data has a much larger importance in this problem than feedling with the model's hyper-parameters or processing of images. If you don't have the right data, the model just won't work as you expected. This proves that mse is not a good performance indicator of the model. Actually trying it out in the simulator is the best way to judge performance.
