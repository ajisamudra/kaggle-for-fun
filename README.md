# Kaggle for fun
All submissions for Kaggle competitions that I have been, and going to be participating

----
## Product Similarity (Image & text data)
[Notebook](https://www.kaggle.com/ajisamudra/product-matching-final-inference-densedistance?scriptVersionId=48253344)

**Problem statement** : Given image and title of two items, predict whether the two items are similar or not.

**Type** : Binary classification (1: match; 0: not match)

**Evaluation** : F1 score

**Additional context** : `there were two test sets`
1. the first was given before the real competiton; it had 207 samples;
2. the second was given in the real competition which only last for 4 hours; it had 32.6k samples; 

**My score** :

* 1st model concatenation embedding of 2 images from DenseNet201 get 0.71493 in first test set
* 2nd model concatenation embedding of 2 images from Xception + label smoothing get 0.88111 in first test set
* 3rd model concatenation embedding of 2 images from DenseNet201 + LR scheduler get **1.0 F1 score in first test set, BUT 0.69018 F1 score in second test set**
* final model using distance features (Cosine, Manhattan, etc) of 2 images using embedding from DenseNet201 get 0.88370 F1 score in first test set, **0.79877 F1 score in second test set**

**Learnings** :
1. Experimenting with image embeddings for similarity learning; the more dense representation the better, that's why we choose DenseNet201 over Xception as the final model
2. Experimenting with label smoothing to regularize the model; it helps when the label is noisy
3. Learn the hard way about overfitting model to the public leaderboard in the first test set
4. In **similarity learning**, we need to build features that represent how different/similar two objects are, not only the concatenation of two objects

----
## Product Classification (Image data)

[Notebook](https://www.kaggle.com/ajisamudra/shopee-object-detection-tpu-efficientnetb6?scriptVersionId=37906178)

**Problem statement** : Predict product class given noisy-colorful image (the images have no standard size).

**Type** : Multi-class classification (42 classes, imbalanced target distribution)

**Evaluation** : top-1 Accuracy

**My score** :

* 1st baseline model using simple transfer learning Inception Resnet v2 get 0.678 Accuracy
* 2nd model using data augmentation and transfer learning Xception get 0.714 Accuracy
* 3rd model using Xception with longer training epoch get 0.780 Accuracy
* 4th model using EfficientNetB4/5 with scheduled learning rate (LR) get 0.818 Accuracy
* 5th model using EfficientNetB6 with scheduled LR + drop small images get **0.823 Accuracy**

----
## Sentiment Classification (Text data)

[Notebook](https://www.kaggle.com/ajisamudra/rating-clf-preprocessed)

**Problem statement** : Predict rating for given review text (the text is in English and might have Bahasa slang).

**Type** : Multi-class classification (5 classes, imbalanced target distribution)

**Evaluation** : top-1 Accuracy

**My score** :

* 1st baseline model using plain CountVectorizer and plain Logistic Regression get 0.407 Accuracy
* 2nd model using preprocessed text, plain TFIDF Vectorizer and plain Logistic Regression get 0.412 Accuracy
* 3rd model using preprocessed text, and FastText model get 0.412 Accuracy
* 4th model using preprocessed text, text augmentation, TFIDF Vectorizer and tuned Logistic Regression get **0.435 Accuracy**

----
## Marketing Response Classification (Tabular data)

[Notebook](https://www.kaggle.com/ajisamudra/shopee-marketing-logreg)

**Problem statement** : Predict user response to marketing email, given both user&email information and user historical behavior in app.

**Type** : Binary classification (1: open, 0: didn't open, imbalanced target distribution)

**Evaluation** : Matthews correlation coefficient (MCC)

**My score** :

* 1st baseline model simple imputation (no user information) and plain LGBM get 0.4969 MCC
* 2nd model using simple feature engineering in user information, dropping unnecessary features, plain LGBM get 0.5218 MCC
* 3rd model using more feature engineering, selecting the significant features, and tuned Logistic Regression get **0.5281 MCC**

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

**Type** : Binary classification (1: real disaster, 0: fake, imbalanced target distribution)

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

**Type** : Binary classification (imbalanced target distribution)

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