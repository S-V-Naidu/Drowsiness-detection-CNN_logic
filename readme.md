This is a project for drowsiness detection.

This uses haarcascade model to local the eye region and feed the eye pixel values to the trained model.

Due to the surronding nature where the camera will be located, if the brightness is too much while training and in prediction it is of low brightness, the values differs. To overcome this limitation, we convert the image to gray scale and to get fine-tune furthermore, we can take the edge values of the image.
The edge method used here is canny edge detection which takes 2 thresholds (min adn max) for the pixel values and filters accordingly.

Now the models are trained with different variations.
1. The whole face is used for training the model and while prediction, the whole face is fed to determine the classes. 
2. Only the eye region of the face is taken and trained for that. Same done for prediction too.

This model works efficiently with 97% accuracy.


