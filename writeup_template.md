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
[image7]: ./examples/sheared.jpg "Flipped Image"
[image7]: ./examples/cropped.jpg "Flipped Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

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

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
