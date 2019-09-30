# ow_ml
IM931 FINAL PROJECT

This is a final project of course IM931, the main goal of the project is to build two model for identifying heroes and maps(Failed) in Overwatch game and reaching over 80% accuracy of this two models when predicting.

The methodology is capturing screenshots of game and chop the weapon icon of the screenshot, which displayed at the right bottom corner to identify the hero that player are using.This is because that the weapon of every hero is unique.
And so as for map identification, chop a fixed part of screenshot and train the model.

Center theory:

The models I built are based on the classic convolutional neural networks: LeNet-5.
LeNet-5  was one of the earliest convolutional neural networks, completed by Yann LeCun in 1994, which promoted the development of deep learning. LeNet-5 used features such as convolution, parameter sharing, and pooling to extract features, full connection avoiding a lot of computational costs and finish classification and identification.


Some difficulties:

The icon of weapon displayed at the right bottom corner is half-transparent. So the color can be changed by the background.
The icon may has a slight up and down vibration as the character jumps, although the position of weapon icon is relatively fixed.
The size of weapon icon

Results:

After using Python to train the model, the accuracy of train and validation have reached closed to 100% when the epoch reaching 10. After the epoch of 10, there are not obvious increase in accuracy or decline in loss function. At the epoch of 40, the accuracy of train has over 99% while the accuracy of validation is over 97%. It is high enough to make a good prediction of which hero by using this model. Then, I randomly picked some picture in size of 1920*1080 to test the model. It shows very high probability of predicting the right answer.Overall, the LeNet-5 model has success to classify heroes.

As for the map classifier, the model did not show the performance as good as the previous one. The first structure I tried is VGG16 which has a great performance in place identification.However, even though the layers are more than LeNet-5, the result did not show as expectation. With the epoch ranging from 0 to 20, the accuracy and the loss function remain stable. The loss function is over 14, while the accuracy stays away from 100%. Due to the limit of hardware, I cannot run the model in a long epoch.
