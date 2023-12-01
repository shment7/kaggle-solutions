# walmart
https://www.kaggle.com/datasets/yasserh/walmart-dataset


Regression problem with tabular data.

We want to predict the weekly sales given store, date etc.


1. Convert store and holiday_flag features to categorical features and perform one hot encoding.
2. Scale numeric features.
3. Use MLP with 2 hidden layers of size 512.
4. batch size=512, lr=0.01, number of epochs=750, L1 loss.
5. Adam optimizer, lr multiplied by 0.1 after each 250 epochs.

Results: 
Mean absolute error - 77618

Mean absolute percentage error - 7.28%

# pump it up
https://www.kaggle.com/datasets/sumeetsawant/pump-it-up-challenge-driven-data?rvi=1


Classification problem with tabular data.

We want to predict if a pump is faulty (functional, non functional, functional needs repair) given information about the pump.


1. Convert district_code and region_code features to categorical features.
2. Remove features wpt_name, installer, funder, subvillage, ward, scheme_name and recorded_by, since they had a lot of missing values or unique categorical values.
3. Convert some of the missing values to np.nan, since the value in the dataset is not consistent (0, 'None', 'unknown').
4. Impute missing values - median for numeric features and most common value for categorical features.
5. Perform one hot encoding.
6. Use xgboost.
7. Hyperparameters were chosen with optuna to minimize cross entropy loss.

   
Results:
Accuracy - 80.8%

# intel image classification
https://www.kaggle.com/datasets/puneet6060/intel-image-classification


Classification problem with images.

We want to predict the image natural scene (buildings, forest, glacier, mountain, sea or street) given the image.


1. Use EfficientNet V2 with pretrained weights that were trained on ImageNet with additional hidden layer at the end to reduce output dimension to 6. All weights are learnable.
2. batch size=64, lr=0.001, number of epochs=7, cross entropy loss, AdamW optimizer.
3. Perform simple augmentation on the images.

Results:
Accuracy - 93%
