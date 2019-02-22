# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center1.jpg "Center Driving"
[image2]: ./examples/bridge1.jpg "Bridge Training"
[image3]: ./examples/bridge2.jpg "Bridge Approach"
[image4]: ./examples/dirt1.jpg "Dirt Recovery"
[image5]: ./examples/dirt2.jpy "Dirt Avoidance"
[image6]: ./examples/recover1.jpg "Normal Image"
[image7]: ./examples/recover2.jpg "Hard Right Recovery"
[image8]: ./examples/recover3.jpg "Hard Right Recovery from Right"
[image9]: ./examples/recover4.jpg "Hard Right Recovery from Left"


## Rubric Points
### Details for the project [rubric points](https://review.udacity.com/#!/rubrics/432/view)
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* *model.py* containing the script to create and train the model
* *drive.py* for driving the car in autonomous mode
* *model16.h5* containing a trained convolution neural network
* *writeup_behavioral_cloning.md* summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator, Udacity provided drive.py file, and my *model16.h5*, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolution neural network.
The file structure is setup as a class that can be executed from the command line - albeit the pipeline is hard-coded. This allowed me to build each step independently, keep track of the test models, and morph the pipeline from a simple test to the final project.

The file pipeline is setup as follows - starting *model.py (ln212)*
1. instantiate the Driver Class
2. collect all the training data and add to the model *ln216-ln264* using *Driver.pre_open_csv()* found on *ln41*. This method collects lines from the csv file.
3. at *ln267* invoke the *Driver.pre_collect_data_gen_help()* method found on *ln107*. This splits the data set into training and validation sets and sets up the generator
4. on *ln269* trigger the model *Driver.nvidia_like()* found on *ln184* which will setup the model
5. on *ln272* compile and run the model via the method *Driver.compile_generator_model()*. This method will also save the model to file.



### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model normalizes the data and crops it *ln193-194*. It then employs a number of convolution neural network with 2x2, 5x5, 5x5, 3x3, 3x3 filter sizes and depths between 24 to 64, Each Conv2D includes RELU layers. *ln198-203*.  The network is then flattened and adds a set of fully connected layers from 100 to 1 *ln204-209*

#### 2. Attempts to reduce overfitting in the model

Two dropout layers were added in order to reduce overfitting *ln195* & *ln205*. I found the 2 dropout layers at .4 (which in keras means an image has 60% chance of being kept) seemed to work well. (although this  was not exhaustively tested).

The model was trained and validated on a wide variety of captured sequences. After each test that resulted in failure, I refined the data set by adding training data specific to the problem points.


#### 3. Model parameter tuning

The model used an adam optimizer *ln121* (all other test models also used the adam optimizer)

I stayed with around 12 epochs. The final model had a loss of *0.0152* and a val_loss of *0.173*

#### 4. Appropriate training data

The model was trained using data of my driving the simulated vehicle on the road. I became better at driving over time (So, maybe I should have thrown out the first awkward training sequences? I kept them in anyway.) I only used the center image data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed an approach similar to the one laid out in the lesson. I wanted to start simply and prove the end-to-end pieces of the pipeline. To that end, I started with a simple class and method that read the csv files and processed them. I added the basic model *ln130* and caused it to generate a file. I then tried it out on the simulator - I think the car ended up in the water.

The next step was to try a LeNet - like approach *ln141*. The result was a little better - it got through the first curve! but did not make it over the bridge.

I tried the nvida like model *ln184*. It worked fairly well but ran off the road at the dirt curve after the bridge. I added more data and then eventually added the dropouts. Along with this network change, I added the generator *ln107 & ln77*.

One interesting note is that my pre-generator design included inverting (flipping?) the images to double to data and to avoid training with a curve bias (see *ln67 & ln68*). When writing this document, I found that the _generator *ln77* does not seem to flip the images. This was would have been an easy add to the generator by inserting the equivalent of the '2 lines of flip code' after *ln97*.


#### 2. Final Model Architecture

The final model architecture *ln184* consisted of a convolution neural network with the following layers and layer sizes ...

| Type                  |     Contents                                |
|:---------------------:|:-------------------------------------------:|
| Normalize             | Lambda(lambda x: (x / 255.0) - 0.           |
| Cropping              | top cropped = 70, bottom crop =25           |
| Dropout               |  0.40                                       |
| Conv2D                | 24, kernel=(2,2), strides=(2,2), "relu"     |
| Conv2D                | 36, kernel=(5,5), strides=(2,2), "relu"     |
| Conv2D                | 48, kernel=(5,5), strides=(2,2), "relu"     |
| Conv2D                | 64, kernel=(3,3),  "relu"                   |
| Conv2D                | 64, kernel=(3,3),  "relu"                   |
| Flatten               |                                             |
| Dropout               |  0.40                                       |
| Dense                 |  100                                        |
| Dense                 |  50                                         |
| Dense                 |  10                                         |
| Dense                 |  1                                          |



#### 3. Creation of the Training Set & Training Process

To capture driving behavior (it took a few tries for it to be good). I recorded a lap and tried to be in the center as best as possible. Here is an example:

![alt text][image1]

When the first training set failed by either heading off the curve or crashing into the bridge, I worked on gathering more data - concentrating on the bridge.
![alt text][image3]
![alt text][image2]

The next problem was with the dirt track after the bridge. The car just kept going. I added data related to the vehicle doing a variety of hard turns away from the dirt.
![alt text][image5]
![alt text][image4]

Then I added recovery data with the vehcle starting close to the edge and moving back to the center. Other data collection included driving the course backwards.

![alt text][image6]
![alt text][image7]
![alt text][image9]
![alt text][image8]


In the end, the final training set consisted of **22422** training images and **5606** validation images. This was the verbose output of the training model.

| Epoch                 |     Training                               |
|:---------------------:|:------------------------------------------:|
| Epoch 1/12            | 54s - loss: 0.0245 - val_loss: 0.0221      |
| Epoch 2/12            | 51s - loss: 0.0216 - val_loss: 0.0212      |
| Epoch 3/12            | 52s - loss: 0.0207 - val_loss: 0.0195      |
| Epoch 4/12            | 51s - loss: 0.0199 - val_loss: 0.0189      |
| Epoch 5/12            | 52s - loss: 0.0193 - val_loss: 0.0190      |
| Epoch 6/12            | 51s - loss: 0.0188 - val_loss: 0.0188      |
| Epoch 7/12            | 52s - loss: 0.0181 - val_loss: 0.0180      |
| Epoch 8/12            | 52s - loss: 0.0175 - val_loss: 0.0184      |
| Epoch 9/12            | 51s - loss: 0.0170 - val_loss: 0.0175      |
| Epoch 10/12           | 51s - loss: 0.0163 - val_loss: 0.0179      |
| Epoch 11/12           | 52s - loss: 0.0158 - val_loss: 0.0172      |
| Epoch 12/12           | 52s - loss: 0.0152 - val_loss: 0.0173      |



