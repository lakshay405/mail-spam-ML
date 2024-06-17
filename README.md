# mail-spam-ML
Email Spam Classification using Logistic Regression
This project aims to classify emails as either spam or legitimate (ham) using machine learning techniques, specifically logistic regression. The classification is based on the content (message body) of the emails.

Dataset
The dataset used in this project (emails.csv) contains email messages labeled as spam or ham. The dataset is preprocessed to handle null values and convert categorical labels into numerical values (0 for spam, 1 for ham).

Workflow
Data Loading and Preprocessing: The email dataset is loaded from emails.csv and null values are replaced with empty strings. The dataset is then split into features (X) containing email messages and labels (Y) indicating spam or ham.

Feature Extraction: Text data (email messages) is transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization. This step converts each email message into a vector representation that is suitable for machine learning algorithms.

Model Training: A logistic regression model is initialized and trained using the TF-IDF transformed features (X_train_features). Logistic regression is chosen for its simplicity and effectiveness in binary classification tasks.

Model Evaluation: The trained model's performance is evaluated on both training and test datasets using accuracy score metrics. This helps in understanding how well the model generalizes to new, unseen data.

Prediction: The trained model is capable of predicting whether a new email message is spam or ham based on its content. An example email is provided as input to demonstrate the prediction functionality.

Example Prediction
An example email message is used to showcase the prediction capability of the model. The model predicts whether the email is likely to be spam or legitimate based on its content.

Technologies Used
Python
Pandas
Scikit-learn (sklearn)
Usage
Ensure Python and necessary libraries (pandas, scikit-learn) are installed.
Clone this repository and navigate to the project directory.
Place your dataset (emails.csv) in the project directory.
Run the script email_classification.py to train the model and perform predictions.
