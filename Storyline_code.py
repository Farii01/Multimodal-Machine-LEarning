# story line
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.svm import SVR

# Send a GET request to the IMDb list URL
url = "https://www.imdb.com/list/ls097459790/"
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Specify the absolute file path to save the scraped data
file_path = "C:/Users/Hp/Desktop/project/Storyline/imdb_data.csv"

# Open the CSV file to save the scraped data
with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Movie', 'Description', 'Rating', 'Genre'])

    # Find all the movie items in the list
    movies = soup.find_all('div', class_='lister-item-content')

    # Iterate over each movie item and extract the desired information
    for movie in movies:
        title = movie.h3.a.text.strip()
        description = movie.find('p', class_='').text.strip()
        rating = movie.find('span', class_='ipl-rating-star__rating').text.strip()
        genre = movie.find('span', class_='genre').text.strip()
        csv_writer.writerow([title, description, rating, genre])

print("Scraping completed successfully and data saved in:", file_path)

# Load the dataset
df = pd.read_csv("C:/Users/Hp/Desktop/project/Storyline/imdb_data.csv")

# Identify the target variables
target_columns = ['Genre', 'Rating']

# Print the preprocessed dataset information
print(df.info())

# Data preprocessing steps
# 1. Missing value omission
df.dropna(inplace=True)

# Perform label encoding on "Movie Names", "Description", and "Genre" (converting them to integers)
label_encoder = LabelEncoder()
df['Movie'] = label_encoder.fit_transform(df['Movie'])
df['Description'] = label_encoder.fit_transform(df['Description'])
df['Genre'] = label_encoder.fit_transform(df['Genre'])

print(df.info())

# visualisation of data to have a better understanding of the relationship
# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Genre', y='Rating')
plt.title('Distribution of Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()

# Visualize the distribution of ratings
sns.histplot(data=df, x='Rating', bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Explanatory data analysis - Visualize correlation using a heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Attribute Correlation Heatmap')
plt.show()

#model training
# Perform label encoding on the target variables
df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Rating'] = label_encoder.fit_transform(df['Rating'])

# Obtain input features (X) and target variables (y) to use in further codes since there is 2 target variables
X = df.drop(['Genre', 'Rating'], axis=1)
y_genre = df['Genre']
y_rating = df['Rating']

# Perform one-hot encoding on categorical features (One-hot encoding is necessary because most machine learning
# algorithms require numerical input data. By one-hot encoding these variables, we create binary features for each
# unique category)
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets for both genre and rating
X_train, X_test, y_genre_train, y_genre_test, y_rating_train, y_rating_test = train_test_split(
    X_encoded, y_genre, y_rating, test_size=0.2, random_state=42)

# Train the Support Vector Regression (SVR) model for rating regression
rating_regressor = SVR()
rating_regressor.fit(X_train, y_rating_train)

# Make predictions on the test set for rating regression
y_rating_pred = rating_regressor.predict(X_test)

# Calculate the R-squared score for rating regression
r2_rating = r2_score(y_rating_test, y_rating_pred)
print("................For genre................ ")
print("R-squared Score (Rating Regression):", r2_rating)

#For rating
# Train the Gradient Boosting Classifier for Rating
gb_rating_classifier = GradientBoostingClassifier()
gb_rating_classifier.fit(X_train, y_rating_train)

# Make predictions on the test set for Rating
y_rating_pred_gb = gb_rating_classifier.predict(X_test)

# Calculate the accuracy of the Gradient Boosting Classifier for Rating
accuracy_rating_gb = accuracy_score(y_rating_test, y_rating_pred_gb)
print("................For rating................ ")
print("Accuracy (Gradient Boosting) for Rating:", accuracy_rating_gb)
