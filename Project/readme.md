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

## APROACHES


## DESCRIPTION
We have used 6 different convolutional filters in this project: 5x5 unweighted, 7x7 unweighted, 9x9 unweighted, weighted 7x7, weighted 5x5, and weighted 9x9 weighted. On each technique, we used these six convolutional filters. Convolution is a pixel level filter effect. It calculates the value of a central pixel by adding all of its peers' weighted values together. The result is a new filtered image that has been changed. First technique that we have applied is multinomialNB this technique is basically optimal for discrete feature classification and we have applied 5 by 5 , 6  by 6 , 7 by 7 weighted and unweighted all these 6 filters on it.We have achieved this multinomialNB technique with convolution filters by  making the 2D convolve function which is basically used for image processing for feature extraction we've looked at an image as a matrix with numbers ranging from 0 to 255 as its components. The dimensions of this matrix are (image height) x (image width) x (image height) (image channels). Second technique we have used is SVM(support vector machine) it attempts to find a boundary that separates the data in a way that minimises the prediction errors and both dense (numpy.ndarray and numpy.asarray) and sparse (any scipy.sparse) sample vectors are supported as input by scikit-learn's SVM.We have achieved this technique by using from sklearn import SVM library basically this library is effective in high dimensional spaces. It is memory effective since it uses a subset of training points in the decision function.Secondly we have make a function of convolution which is use here to perform convolution on 5 by 5 , 7 by 7 and 9 by 9 filters.This function is use for getting filter dimentions for images in SVM technique.In SVM cross validation scores we got are: 0.90 at weighted 5x5 , 0.967 at unweighted 5x5 , 0.89 at weighted 7 by 7 , 0.891 at unweighted 7 by 7 , 0.88 at weighted 9 by 9 , 0.86 at unweighted 9 by 9.

Cross validation score of each technique with screenshots are following:
5 by 5 SVM unweighted convolution filter
![linear 5 by 5 svm unweighted](https://user-images.githubusercontent.com/60998648/115858983-0518a080-a449-11eb-8fba-17cdc0f7d208.PNG)

9 by 9 multinomialNB weighted convolution filter
![multinomialnb 9 by 9 weighted](https://user-images.githubusercontent.com/60998648/115859314-748e9000-a449-11eb-9a36-942cf39fac0b.PNG)

In MultinomialNB.py file we have make check object where we have instantiate MultinomialNB() class we have use fit method which fit naïve bayes classifier according to X,Y(sXTrain, yTrain).In this for loop (for img in X[0:subset,:]) we have convolve all the images do image processing in it.

To tweak the parameters of classifier we have try to stick to simple conventions and keep the number of methods an object must follow to a bare minimum.In scikit-learn, two basic approaches to parameter search are given, one of which we use is GridSearchCV.CNN according to us can be one of the other tecnhnique to get accuracy of 99% on kaggle.

## ACCURACY
We have got 0.96700 accuray on 42000 data using 5x5 SVM unweighted filter.
![output image](https://user-images.githubusercontent.com/60998648/115859740-09918900-a44a-11eb-9025-134471a80205.jpeg)

## REFERENCES



