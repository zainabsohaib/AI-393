### PROJECT MEMBERS
StdID | Name
------------ | -------------
**63726** | **Zainab sohaib** <!--this is the group leader in bold-->
63669 | Sabahat
63555 | Muhammad umair hassan

## PROBLEMS FACES

### Problem 1: 
The first problem we ran into was that when searching for convolution, we came across convolution neural networks, which were solved with deep learning using tenserflow and keras, but after using the tensorflow library to apply tencique, we discovered that we needed to use scipy convolve python.

### Problem 2: 
Second, we spent four days searching for and using CNN only to find later that we wanted to use scipy and numpy instead.

### Problem 3:
We first time worked on datasets before that we did not even know about what Kaggle is. It was very hard for us to read Csv files apply all techniques on that along with our other projects. 

## APROACHES
We have use these following 4 techniques:
* SVM
* Linear Regression
* KNN
* MultinomialNB

## DESCRIPTION
## Convolution:
We have used 6 different convolutional filters in this project: 5x5 unweighted, 7x7 unweighted, 9x9 unweighted, weighted 7x7, weighted 5x5, and weighted 9x9 weighted. On each technique, we used these six convolutional filters. Convolution is a pixel level filter effect. It calculates the value of a central pixel by adding all of its peers' weighted values together. The result is a new filtered image that has been changed. 
## ALL TECHNIQUES DESCRIPTION 
First technique that we have applied is multinomialNB this technique is basically optimal for discrete feature classification and we have applied 5 by 5 , 6  by 6 , 7 by 7 weighted and unweighted all these 6 filters on it.We have achieved this multinomialNB technique with convolution filters by  making the 2D convolve function which is basically used for image processing for feature extraction we've looked at an image as a matrix with numbers ranging from 0 to 255 as its components. The dimensions of this matrix are (image height) x (image width) x (image height) (image channels).A color image has three channels, while a grayscale image has just one (for an RGB).Then convolution on all images is performed where convolutional filtering is performed for  changing an image's spatial and spectral features.We have basically converted the code provided on google classroom of multinomialNB into further remaining techniques to get the accuracy.The best accuracy that we have got is 0.96700 on 5 by 5 unweighted SVM. In multinomialNB the cross validation scores we got are: 5x5 at unweighted filter is 0.81 , 0.82 at 5x5 weighted , 0.76 at 7 by 7 weighted filter , 0.73 at 7 by 7 unweighted filter , 0.74 at weighted 9 by 9 filter and 0.7 at unweighted 9 by 9 filter.Second technique we have used is SVM(support vector machine) it attempts to find a boundary that separates the data in a way that minimises the prediction errors and both dense (numpy.ndarray and numpy.asarray) and sparse (any scipy.sparse) sample vectors are supported as input by scikit-learn's SVM.We have achieved this technique by using from sklearn import SVM library basically this library is effective in high dimensional spaces. It is memory effective since it uses a subset of training points in the decision function.Secondly we have make a function of convolution which is use here to perform convolution on 5 by 5 , 7 by 7 and 9 by 9 filters.This function is use for getting filter dimentions for images in SVM technique.In SVM cross validation scores we got are: 0.90 at weighted 5x5 , 0.967 at unweighted 5x5 , 0.89 at weighted 7 by 7 , 0.891 at unweighted 7 by 7 , 0.88 at weighted 9 by 9 , 0.86 at unweighted 9 by 9.Third technique that we used is KNN ( k-nearest neighbors) it is use for both classification and regression problems. This technique is achieved by using KNeighborsClassifier library where by passing the number of neighbours as an argument to the KNeighborsClassifier( ) function, we have  construct a KNN classifier object.In KNN cross validation scores we got are: 0.77 at 9x9 weighted filter , 0.75 at unweighted filter , 0.77 at weighted 7x7 filter , 0.72 at 7x7 unweighted filter , 0.82 at 5x5 weighted filter and 0.83 at unweighted filter.Fourth technique that we have used is Linear regression basically this technique is use to to minimise the residual number of squares between the observed targets in the dataset, fit a linear model with coefficients. This technique is achieved with convolution filters of 9 by 9 and 5 by 5 and 7 by 7 by  implementing linear regression models for which we have import linear regression class and then instantiate it. In linear regression cross validation scores we got are: -1.230 at unweighted 5 by 5 filter , -2.70 at weighted 5x5 filter , -3.63 at 7 by 7 weighted filter , -2.49 at 7 by 7 weighted filter , -30.90 at 9 by 9 weighted filter , -89.00 at 9x9 un weighted filter. 

Cross validation score of each technique with screenshots are following:
* 5 by 5 SVM unweighted convolution filter
![linear 5 by 5 svm unweighted](https://user-images.githubusercontent.com/60998648/115858983-0518a080-a449-11eb-8fba-17cdc0f7d208.PNG)

* 9 by 9 multinomialNB weighted convolution filter
![multinomialnb 9 by 9 weighted](https://user-images.githubusercontent.com/60998648/115859314-748e9000-a449-11eb-9a36-942cf39fac0b.PNG)

* 5 by 5 linear regression unweighted filter
![linear unweighted 5 by 5](https://user-images.githubusercontent.com/68737826/115860964-a99be200-a44b-11eb-8f10-8232e4560bf3.PNG)

* 5 by 5 multinomialNB unweighted filter
![multinomialNB 5 by 5 unweighted](https://user-images.githubusercontent.com/68737826/115861147-df40cb00-a44b-11eb-9172-82c5c9855e73.PNG)

* 7 by 7 KNN weighted filter
![knn 7 by 7 weighted](https://user-images.githubusercontent.com/62794527/115861789-ace39d80-a44c-11eb-941e-37c90b46654f.PNG)

* 9 by 9 KNN weighted filter
![knn weighted 9 by 9](https://user-images.githubusercontent.com/62794527/115861797-b10fbb00-a44c-11eb-9f04-f5fcae823034.PNG)


In .py files we have use 2D array to store value of filters where ([[1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1]]) is use for  unweighted convolution filter , ([[1,1,1,1,1,1,1,1,1], [1,2,2,2,2,2,2,2,1], [1,2,3,3,3,3,3,2,1], [1,2,3,4,4,4,3,2,1], [1,2,3,4,5,4,3,2,1], [1,2,3,4,4,4,3,2,1], [1,2,3,3,3,3,3,2,1], [1,2,2,2,2,2,2,2,1], [1,1,1,1,1,1,1,1,1]]) weighted filter.
In MultinomialNB.py file we have make check object where we have instantiate MultinomialNB() class we have use fit method which fit naïve bayes classifier according to X,Y(sXTrain, yTrain).In this for loop (for img in X[0:subset,:]) we have convolve all the images do image processing in it.In LinearRegression.py file after instantiating linear regression class fit method is call with our train to train on our data. (sXTest) contains predicted values for input values of numpy array.In KNN.py file KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean') finds the k neighbors of a point and below it fit method fits the k-nearest neighbors classifier from the training dataset. Predict(sXTest) is predicting the target for provided data.In SVM.py fit method is fitting the SVM model according to given training data and predict(sXTest) is performing classification on samples in X.To tweak the parameters of classifier we have try to stick to simple conventions and keep the number of methods an object must follow to a bare minimum.In scikit-learn, two basic approaches to parameter search are given, one of which we use is GridSearchCV.CNN according to us can be one of the other tecnhnique to get accuracy of 99% on kaggle.

## ACCURACY
We have got 0.96700 accuray on 42000 data using 5x5 SVM unweighted filter.
![output image](https://user-images.githubusercontent.com/60998648/115859740-09918900-a44a-11eb-9025-134471a80205.jpeg)
![accuracy](https://user-images.githubusercontent.com/68737826/115861253-05ff0180-a44c-11eb-9cd7-c46eba86e6fd.jpeg)

## REFERENCES
* https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
* https://stackabuse.com/linear-regression-in-python-with-scikit-learn/



