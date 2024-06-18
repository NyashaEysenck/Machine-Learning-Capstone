"""
This module provides functions for generating personalized course recommendations
based on user profiles and course data.

Functions:
- user_profile_recommendations(user, score_threshold=14): 
    Generates recommendation scores for users based on their profiles and filters 
    courses based on a score threshold.

Imports:
- From utils.data_loader:
    - load_ratings: Function to load user ratings.
    - load_course_genres: Function to load course genres.
- From utils.preprocessor:
    - get_user_vector: Function to retrieve the user profile vector.
- External imports:
    - numpy as np: Numerical operations library.
"""

from utils.data_loader import load_ratings, load_course_genres
from utils.preprocessor import get_user_vector
import numpy as np

def user_profile_recommendations(user, score_threshold = 14):
    """
    Generate recommendation scores for users and courses.

    Returns:
    users (list): List of user IDs.
    courses (list): List of recommended course IDs.
    scores (list): List of recommendation scores.
    """

    users = []      # List to store user IDs
    courses = []    # List to store recommended course IDs
    scores = []     # List to store recommendation scores

 
    ratings_df = load_ratings()
    course_genres_df = load_course_genres()
    all_courses = set(course_genres_df['COURSE_ID'].values)

    # Get the user profile data for the current user
    test_user_profile = get_user_vector(user)

    # Get the user vector for the current user id (replace with your method to obtain the user vector)
    test_user_vector = test_user_profile.values.flatten()
    #print(test_user_vector)

    # Get the known course ids for the current user
    enrolled_courses = ratings_df[ratings_df['user'] == user]['item'].to_list()

    # Calculate the unknown course ids
    unknown_courses = all_courses.difference(enrolled_courses)

    # Filter the course_genres_df to include only unknown courses
    unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].values

    # Calculate the recommendation scores using dot product
    recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

    # Append the results into the users, courses, and scores list
    for i in range(0, len(unknown_course_ids)):
        score = recommendation_scores[i]

        # Only keep the courses with high recommendation score
        if score >= score_threshold:
            users.append(user)
            courses.append(unknown_course_ids[i])
            scores.append(recommendation_scores[i])
 
    return users, courses, scores