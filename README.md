# Multimodal_Machine_Learning
Steps taken:
1.	Data Collection by Web scrapping : Web scraping tools were employed to collect the necessary data from IMDB. I learned this new way of collecting information for a dataset, which I didn’t know about before. I scrapped the information from the html code of the website which was quite challenging.

2.	Data pre-processing: Each dataset was properly cleaned and necessary steps were taken to prepare them for their model training. Applied scaling, feature extraction, conversion and other important steps where it was necessary.

3.	Exploratory Data Analysis (EDA): An exploratory data analysis was conducted to gain insights into the dataset. Key aspects explored in each code to have a visual representation of the attributes and their correlation. These analyses provided a better understanding of the dataset's characteristics.

4.	Model Building: Different models were used for different datasets depending on the nature of the dataset.

5.	Model fusion: Building separate models for each feature category helped me to capture the distinct characteristics and patterns present in different types of data. By combining the predictions of these models, I created a comprehensive system that provides accurate and robust predictions for movie ratings and genres.

Output:
According to the outcome of the fusion model, the model's performance in predicting movie genres is moderate, achieving an accuracy of 50%. This means that the model correctly identifies the genre of about half of the movies in the test set. While there is room for improvement, it shows that the model is capturing some patterns and characteristics related to different genres.
On the other hand, the model's performance in predicting movie ratings is remarkable. All of the predicted ratings are within a 10% range of the actual ratings, resulting in a perfect prediction rate of 100%. This indicates that the model is highly accurate in estimating the ratings of movies based on the provided information. It suggests that the model has successfully learned the relationships and factors that contribute to the ratings, resulting in precise estimations.
