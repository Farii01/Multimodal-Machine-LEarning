# categorical

from sklearn.svm import SVR
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import csv
from sklearn.svm import SVC

# Send a GET request to the IMDb list URL
url = "https://www.imdb.com/list/ls097459790/"

# Create a BeautifulSoup object to parse the HTML content
response = requests.get(url)

# Specify the absolute file path to save the scraped data
soup = BeautifulSoup(response.content, 'html.parser')

# Specify the absolute file path to save the scraped data
file_path = "C:/Users/Hp/Desktop/project/Categorical/movie_dataset.csv"

# Open the CSV file to save the scraped data
with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Movie', 'Director', 'Writer', 'Rating', 'Genre'])

    # Find all the movie items in the list
    movies = soup.find_all('div',
                           class_='lister-item-content')  # html tags are used to find the items from the html code of the website

    # Iterate over each movie item and extract the desired information
    for movie in movies:
        title = movie.h3.a.text.strip()
        # if the specific html tag is not found, it will be filled with "unknown"
        director_element = movie.find('p', class_='text-muted text-small')
        director = "Unknown Director"
        if director_element:
            director_link = director_element.find('a')
            if director_link:
                director = director_link.text.strip()

        writer_elements = movie.find_all('a')[1:]  # Exclude the first <a> tag (director)
        writer = ", ".join([writer.text.strip() for writer in writer_elements]) if writer_elements else "Unknown Writer"

        rating = movie.find('span', class_='ipl-rating-star__rating').text.strip()
        genre = movie.find('span', class_='genre').text.strip()

        csv_writer.writerow([title, director, writer, rating, genre])

print("Scraping completed successfully and data saved in:", file_path)

# Load the dataset
df = pd.read_csv("C:/Users/Hp/Desktop/project/Categorical/movie_dataset.csv")

# Perform data preprocessing
# Drop any rows with missing values
df.dropna(inplace=True)
print(df)
print(df.info())

# Perform label encoding on categorical variables (convert to integer)
label_encoder = LabelEncoder()
df['Movie'] = label_encoder.fit_transform(df['Movie'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Writer'] = label_encoder.fit_transform(df['Writer'])
df['Rating'] = label_encoder.fit_transform(df['Rating'])
df['Genre'] = label_encoder.fit_transform(df['Genre'])

print("The dataset after converting it to numeric: ")
print(df.info())

# Visual representation of the variables
# Data visualization - Visualize correlation using a heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Attribute Correlation Heatmap')
plt.show()

# Visualize the distribution of ratings
sns.histplot(data=df, x='Rating', bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# Model training
# Split the dataset into input features (X) and target variables (y_genre, y_rating)
X = df.drop(['Genre', 'Rating'], axis=1)
y_genre = df['Genre']
y_rating = df['Rating']

# Split the dataset into training and testing sets
X_train, X_test, y_genre_train, y_genre_test, y_rating_train, y_rating_test = train_test_split(
    X, y_genre, y_rating, test_size=0.2, random_state=42
)

# Train the Support Vector Regression (SVR) model for genre regression
genre_regressor = SVR()
genre_regressor.fit(X_train, y_genre_train)

# Make predictions on the test set for genre regression
y_genre_pred = genre_regressor.predict(X_test)

# Calculate the R-squared score for genre regression
r2_genre = r2_score(y_genre_test, y_genre_pred)
print("................For genre................ ")
print("R-squared Score (Genre Regression):", r2_genre)
print("               ")

# Train the Random Forest Regressor for rating regression
rating_regressor = RandomForestRegressor()
rating_regressor.fit(X_train, y_rating_train)

# Make predictions on the test set for rating regression
y_rating_pred = rating_regressor.predict(X_test)

# Calculate R-squared score and mean absolute error for rating regression
r2_rating = r2_score(y_rating_test, y_rating_pred)
mae_rating = mean_absolute_error(y_rating_test, y_rating_pred)

print("................For rating................ ")
print("R-squared (Rating Regression):", r2_rating)
print("MAE (Rating Regression):", mae_rating)
