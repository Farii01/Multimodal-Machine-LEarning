# numeric
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

# Define the URL of the IMDb webpage containing the movie list
url = "https://www.imdb.com/list/ls097459790/"

# Send an HTTP GET request to the URL and store the response
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content of the response
soup = BeautifulSoup(response.content, 'html.parser')

# Scrape movie names
scraped_movies = soup.find_all('h3', class_='lister-item-header')  # these are html tags where the name of the movie exits
movies = []  # taking empty list to keep all the scrapped names
for movie in scraped_movies:  # a loop is initiated to go through the html code of the website and fetch all the names
    movie_name = movie.a.text.strip()
    movies.append(movie_name)

# Scrape ratings (same as movie names)
scraped_ratings = soup.find_all('div', class_='ipl-rating-star small')
ratings = []
for rating in scraped_ratings:
    rating_element = rating.find('span', class_='ipl-rating-star__rating')
    if rating_element:
        rating_value = rating_element.text.strip()
        ratings.append(float(rating_value))
    else:
        ratings.append(0.0)  # Assign a default rating of 0.0 if the rating is missing

# Scrape genre, duration, and budget  (same as movie names)
scraped_details = soup.find_all('p', class_='text-muted text-small')
genres = []
durations = []
budgets = []
for details in scraped_details:
    genre_element = details.find('span', class_='genre')
    genre = genre_element.text.strip() if genre_element else "Unknown"
    genres.append(genre)

    duration_element = details.find('span', class_='runtime')
    duration = duration_element.text.strip() if duration_element else "Unknown"
    durations.append(duration)

    budget_element = details.find('span', class_='text-muted')
    budget = budget_element.find_next_sibling('span').text.strip() if budget_element else "Not available"
    budgets.append(budget)

# Ensure all lists have the same length
min_length = min(len(movies), len(ratings), len(genres), len(durations), len(budgets))
movies = movies[:min_length]
ratings = ratings[:min_length]
genres = genres[:min_length]
durations = durations[:min_length]
budgets = budgets[:min_length]

# DataFrame to store the scraped data
data = pd.DataFrame({
    'Movie': movies,
    'Rating': ratings,
    'Genre': genres,
    'Duration': durations,
    'Budget': budgets
})

print("All information: ")
print(data.head())
print("                                       ")
print("                                       ")
print(data.info())
print("                                       ")
print("                                       ")

# Data preprocessing steps
data.dropna(inplace=True)  # Drop any rows with missing values

# encoder is used to convert into integer
label_encoder = LabelEncoder()
data['Movie'] = label_encoder.fit_transform(data['Movie'])
data['Genre'] = label_encoder.fit_transform(data['Genre'])
data['Duration'] = label_encoder.fit_transform(data['Duration'])
data['Budget'] = label_encoder.fit_transform(data['Budget'])

# Split the dataset into features and target
X = data[['Movie', 'Genre', 'Duration', 'Budget']]
y_genre = data['Genre']
y_rating = data['Rating']

print("                                       ")
print("                                       ")
print("All information after preprocessing: ")
print(data)
print("                                       ")
print("                                       ")
print(data.info())
print("                                       ")
print("                                       ")

# Data exploration to see relationship
# Visualize the distribution of ratings
sns.histplot(data=data, x='Rating', bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Visualize the relationship between duration and rating
sns.scatterplot(data=data, x='Duration', y='Rating')
plt.title('Duration vs Rating')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.show()

# Visualize the average rating for each genre
sns.barplot(data=data, x='Genre', y='Rating')
plt.title('Average Rating by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# Visualize the average rating for each budget category
sns.boxplot(data=data, x='Budget', y='Rating')
plt.title('Average Rating by Budget')
plt.xlabel('Budget')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.show()

#model training for genre and rating

# Split the dataset into training and testing sets for both genre and rating
X_train_genre, X_test_genre, y_train_genre, y_test_genre = train_test_split(
    X, y_genre, test_size=0.2, random_state=42)
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
    X, y_rating, test_size=0.2, random_state=42)

# Genre
# Train the Random Forest Classifier for genre classification (this is needed for training the Random Forest
# Classifier model and using it to make predictions on new, unseen data.)
genre_classifier = RandomForestClassifier()
genre_classifier.fit(X_train_genre, y_train_genre)

# Make predictions on the test set for genre classification
y_genre_pred = genre_classifier.predict(X_test_genre)

# Calculate the accuracy for genre classification
accuracy_genre = accuracy_score(y_test_genre, y_genre_pred)
print("................For genre................ ")
print("Accuracy for genre (Genre Classification):", accuracy_genre)
print("                                       ")
print("                                       ")

# For rating, I considered both test and training sets
# Initialize and train a Random Forest Regression model
model = RandomForestRegressor()
model.fit(X_train_rating, y_train_rating)

# Make predictions on the training set
y_train_pred = model.predict(X_train_rating)

# Make predictions on the test set
y_test_pred = model.predict(X_test_rating)

# Calculate R-squared score for training and test sets
train_r2 = r2_score(y_train_rating, y_train_pred)
test_r2 = r2_score(y_test_rating, y_test_pred)

# Calculate mean absolute error for training and test sets
train_mae = mean_absolute_error(y_train_rating, y_train_pred)
test_mae = mean_absolute_error(y_test_rating, y_test_pred)
print("................For rating................ ")
print("Train R-squared:", train_r2)
print("Test R-squared:", test_r2)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
