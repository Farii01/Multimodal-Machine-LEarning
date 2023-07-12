import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load data from the three models
data_numeric = pd.read_csv("C:/Users/Hp/Desktop/project/Numeric/numeric_model_data.csv")  # Load numeric model data
data_storyline = pd.read_csv("C:/Users/Hp/Desktop/project/Storyline/imdb_data.csv")  # Load storyline data
data_categorical = pd.read_csv("C:/Users/Hp/Desktop/project/Categorical/movie_dataset.csv")  # Load categorical data

# Numeric data preprocessing
X_numeric = data_numeric.drop(['Genre', 'Rating'], axis=1)  # Extract features for numeric data
y_numeric_genre = data_numeric['Genre']  # Extract genre labels for numeric data
y_numeric_rating = data_numeric['Rating']  # Extract rating labels for numeric data

# Storyline data preprocessing
X_storyline = data_storyline.drop(['Genre', 'Rating'], axis=1)  # Extract features for storyline data
y_storyline_genre = data_storyline['Genre']  # Extract genre labels for storyline data
y_storyline_rating = data_storyline['Rating']  # Extract rating labels for storyline data

# Categorical data preprocessing
X_categorical = data_categorical.drop(['Genre', 'Rating'], axis=1)  # Extract features for categorical data
y_categorical_genre = data_categorical['Genre']  # Extract genre labels for categorical data
y_categorical_rating = data_categorical['Rating']  # Extract rating labels for categorical data

# Combine all the data
X_combined = pd.concat([X_numeric, X_storyline, X_categorical], axis=1)  # Combine numeric, storyline, and categorical features

# One-hot encode categorical variables
categorical_columns = X_categorical.columns  # Get the column names of the categorical features
encoder = OneHotEncoder()  # Create an instance of the OneHotEncoder
X_combined_encoded = encoder.fit_transform(X_combined[categorical_columns]).toarray()  # Perform one-hot encoding on the categorical features

# Split the data into training and testing sets for each model
X_combined_train, X_combined_test, y_numeric_genre_train, y_numeric_genre_test, y_numeric_rating_train, y_numeric_rating_test = train_test_split(
    X_combined_encoded, y_numeric_genre, y_numeric_rating, test_size=0.2, random_state=42)  # Split the combined data into training and testing sets

# Numeric model training
numeric_classifier = RandomForestClassifier()  # Create a random forest classifier for genre prediction
numeric_classifier.fit(X_combined_train, y_numeric_genre_train)  # Train the numeric classifier
numeric_regressor = RandomForestRegressor()  # Create a random forest regressor for rating prediction
numeric_regressor.fit(X_combined_train, y_numeric_rating_train)  # Train the numeric regressor

# Numeric model predictions
y_numeric_genre_pred = numeric_classifier.predict(X_combined_test)  # Make genre predictions using the numeric classifier
y_numeric_rating_pred = numeric_regressor.predict(X_combined_test)  # Make rating predictions using the numeric regressor

# Evaluate the performance of the numeric model - Genre
numeric_genre_accuracy = accuracy_score(y_numeric_genre_test, y_numeric_genre_pred)  # Calculate accuracy of genre predictions

# Calculate the performance of ratings predicted within 10% of the actual rating
rating_difference = abs(y_numeric_rating_test - y_numeric_rating_pred)
within_range = rating_difference <= 0.1 * y_numeric_rating_test
rating_within_range_percentage = (within_range.sum() / len(within_range)) * 100

# Print the performance metrics
print("Numeric Model - Genre Accuracy:", numeric_genre_accuracy)  # Print the accuracy of genre predictions
print("Rating Prediction Performance - Percentage within 10% range:", rating_within_range_percentage)  # Print the percentage of rating predictions within the acceptable range
