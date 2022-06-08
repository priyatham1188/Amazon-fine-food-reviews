# Amazon-fine-food-reviews
The data set consists of 14 features along with the target value which is used to predict whether the person has heart disease or not ,It is a multivariate data set it consists or includes a number of distinct mathematical or statistical variables, as well as multivariate numerical data analysis.

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

Number of reviews: 568,454
Number of users: 256,059
Number of products: 74,258
Timespan: Oct 1999 - Oct 2012
Number of Attributes/Columns in data: 10
Attribute Information: Columns in the dataset

Id : The id for each row of data
ProductId - unique identifier for the product's in amazon fine food category
UserId - A primary identifier for each user
ProfileName - Profile name of the associated user
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
Score - rating which varies between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review

# Question 3

a) Objective: Predicting the sentiment of the given review

I've applied LSTM,Random Forest,KNN , Naive bayes on this data set

Steps Performed on the dataset:

1.  I've aquired this datset from the kaggle thorugh this link https://www.kaggle.com/snap/amazon-fine-food-reviews

2.  DataCleaning:

        I've applied few techniques in order to remove any missing , duplicate values within the dataset .

        As well as per the dataset The helpfulness numerator which means the number of users who found it helpful whcih should be always less than the denominator which includes both kind of users who found it helpful or not we identified those and removed them


        If we observe the data we have reviews whose rating ranges from 1 to 5 however we see there are review with rating 3 which are considered as neutral reviews as we need to predict either positive or negative reviews so I removed these rows with neutral review

        we also encoded the the review greater than 3 as 1 and the ones less than 3 as 0 .

3.  Data Pre-processing:

    In this we performed the activities like Removing Punctuations, html tags from the texts of the review .

    Stopwords: we have removed the stop words like (and,is,are,etc,..) from the text as its doesnt help the model in understanding the sentiment of the user and will also help model to learn the more weighted words within the reviews

    Stemming: We have applied a technique called stemming which stems the words to its root so that all the words which are of common but change in their tense will be treated as one word and will help in reducing the vector dimension and also converted the text into lower case for encoding purpose.

    Encoding the data: I've applied the Bag of words technique in order to encode the text data into a vector form

4.  Observations

    # Accuracies of the classifiers used :

    LSTM: 87.88

    Naive bayes: 86.81

    Random forest: 85.51

    I have ran this pre-processed data on LSTM , Naive bayes, Random forest

    After running all the models it's observed that LSTM outperformed w.r.t to its accuracy on test/validation set .

