# RF-Pathloss-Predictor

## The prediction model
Estimating a path loss can be solved by machine learning techniques to overcome challenging issues such as complexity and time consuming due to the required tremendous measurements.Path loss prediction models are used to estimate the coverage area of a given transmitter.The model was developed using python and machine learning concepts. Here, we have used both linear regression and KNN algorithm to develop our model. Both algorithms are prediction algorithms used in machine learning. 
## Linear regression
Linear regression is a statistical method for studying relationships between an independent variable X and Y dependent variable. It is mathematical modeling which allows you to make predictions for the value of Y depending on the different values of X.

### Example of simple linear regression
The first prediction model was created using the Multiple linear regression in linear regression. Multiple linear regression is a case of linear regression with two or more independent variables. Here Y is the dependent response which is the path loss, X is the independent variable which is in our case distance and frequency.


## K - Nearest Neighbor regression 
K Nearest Neighbor is one of the fundamental algorithms in machine learning. KNN is one of the simplest forms of machine learning algorithms mostly used for classification and regression. It classifies the data point on how its neighbor is classified.K in KNN represents the number of the nearest neighbors we used to classify new data points.
The second prediction model was created using the K Nearest Neighbor regression. KNN regression tries to predict the value of the output variable by using a local average.Here also Y is the dependent response which is the path loss, X is the independent variable which is in our case distance and frequency. 

### Example of simple K - Nearest Neighbor regression
Suppose the prediction point is x.  In the prediction model the number of neighbors needs to be defined and in this case it is 3. The KNN algorithm starts by calculating the distance of point X from all the points. It then finds the 3 nearest points with least distance to point X. Then the average of the values is taken to be the final prediction of the prediction point that input.


A dataset with real data which has the real path loss after calculating it considering the factors which affect the path loss like absorption, diffraction, atmosphere etc. was used to train and test the developed model. In both models the dataset was divided into two parts and 80% of it was used to train the model and the rest of the data was used to test the model. Once the models are created and trained using the dataset, predictions can be done with either existing or new data.The prediction model was connected to the site. Users can enter values for frequency and distance through the site and model will evaluate the inputs and produce an output for path loss.
Mean absolute error (MAE) and Mean squared error (MSE) metrics are used to evaluate these prediction models  Here, the path loss real values from Y dependent variable and the path loss predicted values from X independent variables are used to test and generate the mean absolute error and the mean squared error to determine the accuracy of the models. 
Flask is used to combine the python models and web frontend. It is a popular Python web framework that is used to communicate between the python models and the user via HTTP requests.

## Mean Absolute Error(MAE)
The mean absolute error is the average of the difference between the predicted values and the actual values.

## Mean Squared Error(MSE)
The mean Squared error is the average of the squared differences between the predicted values and the actual values.

