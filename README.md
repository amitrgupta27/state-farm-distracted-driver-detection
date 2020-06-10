# state-farm-distracted-driver-detection
Classification of various distracted driver poses using mask RCNN and ensemble of VGG16, Mobilenet, Resnet50 and Xception models.

The project uses Mask RCNN model to crop the driver image, and uses this image to train an ensemble of VGG16, Mobilenet, Resnet50 and Xception. 

The above approach resulted in a private leaderboard score of 0.25584 on the kaggle state farm distracted driver competition resulting in a top 11% finish.

The project combines the ideas and presented in  https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1 and https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 and leverages the code found in https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4

