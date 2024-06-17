import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data from CSV file into a Pandas DataFrame
raw_email_data = pd.read_csv('emails.csv')
print(raw_email_data)

# Replace null values with empty strings
email_data = raw_email_data.where((pd.notnull(raw_email_data)), '')

# Display the first 5 rows of the dataframe
email_data.head()

# Check the number of rows and columns in the dataframe
print("Shape of the dataframe:", email_data.shape)

# Label spam emails as 0 and ham (non-spam) emails as 1
email_data.loc[email_data['Category'] == 'spam', 'Category'] = 0
email_data.loc[email_data['Category'] == 'ham', 'Category'] = 1

# Separate the data into texts (X) and labels (Y)
X = email_data['Message']
Y = email_data['Category']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Transform the text data into feature vectors using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = tfidf_vectorizer.fit_transform(X_train)
X_test_features = tfidf_vectorizer.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Display examples of X_train and its features
print("Example of X_train:", X_train.iloc[0])
print("Example of X_train features:", X_train_features[0])

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_features, Y_train)

# Predictions on training data
train_predictions = logistic_model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)
print('Accuracy on training data:', train_accuracy)

# Predictions on test data
test_predictions = logistic_model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print('Accuracy on test data:', test_accuracy)

# Example input mail for prediction
input_email = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# Convert input text to feature vectors
input_email_features = tfidf_vectorizer.transform(input_email)

# Make predictions
prediction = logistic_model.predict(input_email_features)
print(prediction)

# Output the prediction result
if prediction[0] == 1:
    print('Predicted: Legitimate email')
else:
    print('Predicted: Spam email')
