# Kaggle for fun
All submissions for Kaggle competitions that I have been, and going to be participating

----
## Digit Recognizer

[Notebook](https://www.kaggle.com/ajisamudra/solving-mnist-using-cnn-mlp-and-stacking)

**Problem statement** : Predict digit value given handwriting picture (28 x 28 pixel).

**Type** : Multi-class classification (0-9 digit)

**Evaluation** : Accuracy

**My score** :

* 1st baseline model using Logistic Regression get 0.956 Accuracy
* 2nd model using ensemble of binary-classifier Logistic Regression get 0.924 Accuracy
* 3rd model using Multi-Layer Perceptron / Fully Connected Neural Network get 0.977 Accuracy
* 4th model using Convolutional Neural Network (CNN) get 0.9851 Accuracy
* 5th model using tuned Convolutional Neural Network (CNN) get **0.9910 Accuracy**

----
## NLP with Disaster Tweet

[Notebook](https://www.kaggle.com/ajisamudra/nlp-count-tf-idf-hashing-vectorizer)

**Problem statement** : Predict whether the tweet represent real disaster or not given the text of tweet.

**Type** : Binary classification

**Evaluation** : F1 score

**My score** :

* 1st baseline model using Logistic Regression and Count Vectorizer get 0.7354 F1 score
* 2nd model using Logistic Regression and Hashing Vectorizer get 0.7401 F1 score
* 3rd model using Logistic Regression and TF-IDF Vectorizer get 0.7466 F1 score
* 4th model using Fully Connected Neural Network and TF-IDF Vectorizer get 0.72790 F1 score
* 5th model using Fully Connected Neural Network and Count Vectorizer get 0.73542 F1 score
* 6th model using Logistic Regression and adjusted hyperparameter Count Vectorizer get **0.79856 F1 score**

----
## Categorical Feature Encoding Challenge

[Notebook](https://www.kaggle.com/ajisamudra/modelling-with-categorical-features)

**Problem statement** :  Get highest evaluation metric given several categorical features.

**Type** : Binary classification

**Evaluation** : AUC

**My score** :

* 1st baseline model using Logistic Regression (only label encoding) get 0.70898 AUC
* 2nd model using Logistic Regression (log & power transformation) get 0.7339 AUC
* 3rd model using Logistic Regression (log & power transformation; target encoding + one-hot encoding) get 0.77515 AUC
* 4th model using Ensemble of LGBM, CatBoost, and Logistic Regression get **0.78273 AUC**

----
## House Price Prediction

[Notebook](https://www.kaggle.com/ajisamudra/house-price-prediction)

**Problem statement** : Predict house price given many features which describe house condition and residential area.

**Type** : Regression

**Evaluation** : RMSE

**My score** :

* 1st model using Random Forest get **0.14405 RMSE**

----
## Titanic Machine Learning from Disaster

[Notebook](https://www.kaggle.com/ajisamudra/survivor-of-titanic-prediction)

**Problem statement** : Predict who survived given passengers information.

**Type** : Binary classification

**Evaluation** : Accuracy

**My score** :

* 1st model using Random Forest get **0.79425 Accuracy**

----
## Pokemon Combat Winner Prediction

[Notebook](https://www.kaggle.com/ajisamudra/winner-in-pokemon-combat-prediction)

**Problem statement** : Spot machine learning opportunity using Pokemon combat dataset. Predict winner pokemon given battle between two pokemons information.

**Type** : Binary classification

**Evaluation** : Accuracy

**My score** :

* 1st model using Decision Tree get 0.9435 Accuracy
* 2nd model using Random Forest get **0.9738 Accuracy**