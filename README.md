# Walmart
https://www.kaggle.com/datasets/yasserh/walmart-dataset


Regression problem with tabular data.

We want to predict the weekly sales given store, date etc.
Data consists of 7 features and 6435 samples - 90% in training set and 10% in testing set.

1. Convert store and holiday_flag features to categorical features and perform one hot encoding.
2. Scale numeric features.
3. Use MLP with 2 hidden layers of size 512.
4. batch size=512, lr=0.01, number of epochs=750, L1 loss.
5. Adam optimizer, lr multiplied by 0.1 after each 250 epochs.

Results: 

Mean absolute error - 77618

Mean absolute percentage error - 7.28%

# Pump It Up
https://www.kaggle.com/datasets/sumeetsawant/pump-it-up-challenge-driven-data?rvi=1


Classification problem with tabular data.

We want to predict if a pump is faulty (functional, non functional, functional needs repair) given information about the pump.
Data consists of 40 features and 59,400 samples - 90% in training set and 10% in testing set.
54% of the data labeled as functional, 39% as non functional and 7% as functional needs repair.

1. Convert district_code and region_code features to categorical features.
2. Remove features wpt_name, installer, funder, subvillage, ward, scheme_name and recorded_by, since they had a lot of missing values or unique categorical values.
3. Convert some of the missing values to np.nan, since the value in the dataset is not consistent (0, 'None', 'unknown').
4. Impute missing values - median for numeric features and most common value for categorical features.
5. Perform one hot encoding.
6. Use XGBoost.
7. Hyperparameters were chosen with optuna to minimize cross entropy loss with 5 folds cross validation.

   
Results:

Accuracy - 80.8%

# Intel Image Classification
https://www.kaggle.com/datasets/puneet6060/intel-image-classification


Classification problem with images.

We want to predict the image natural scene (buildings, forest, glacier, mountain, sea or street) given the image.
Data consists of 25,000 RGB images of size 150x150 - 90% of it in training set and 10% in testing set.
The data is balanced.

1. Use EfficientNet V2 with pretrained weights that were trained on ImageNet with additional hidden layer at the end to reduce output dimension to 6. All weights are learnable.
2. batch size=64, lr=0.001, number of epochs=7, cross entropy loss, AdamW optimizer.
3. Perform simple augmentations(horizontal flip, blur, etc) on the images.

Results:

Accuracy - 93%

# IMDb Movie Reviews
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


Sentiment analysis problem with text.

We want to predict the sentiment of a movie review (positive or negative) given the review.
Data consists of 50,000 reviews - 90% of it in training set and 10% in testing set.
25,000 reviews with positive sentiment and 25,000 reviewswith negative sentiment.

1. Use DeBERTa V3 base with pretrained weights with classification head on top. All weights are learnable.
2. batch size=8, lr=1e-5, number of epochs=2, cross entropy loss, AdamW optimizer.
3. Perform simple augmentations(swap some of the words, apply spelling error to some words) on the reviews.

Results:

Accuracy - 96%
