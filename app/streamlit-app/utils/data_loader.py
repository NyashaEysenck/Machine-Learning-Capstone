"""
This module provides functions to load various datasets required for the course recommendation system.

Functions:
- load_ratings(): Loads user ratings from 'data/ratings.csv'.
- load_user_profiles(): Loads user profiles from 'data/user_profile.csv'.
- load_course_sims(): Loads course similarity data from 'data/sim.csv'.
- load_courses(): Loads processed course data from 'data/course_processed.csv' and titles are capitalized.
- load_course_genres(): Loads course genre data from 'data/course_genre.csv'.
- load_bow(): Loads bag-of-words representations of courses from 'data/courses_bows.csv'.
- add_new_ratings(new_courses): Adds new user ratings to 'data/ratings.csv' based on provided courses.

Imports:
- pandas as pd: Data manipulation library.
"""
import pandas as pd

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          )

def load_ratings():
    return pd.read_csv("data/ratings.csv")

def load_user_profiles():
    return pd.read_csv('data/user_profile.csv')

def load_course_sims():
    return pd.read_csv("data/sim.csv")

def load_courses():
    df = pd.read_csv("data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_course_genres():
    return pd.read_csv("data/course_genre.csv")

def load_bow():
    return pd.read_csv("data/courses_bows.csv")

def add_new_ratings(new_courses):
    """
    Adds new user ratings to 'data/ratings.csv' based on provided courses.

    Args:
    new_courses (list): List of new course IDs to add ratings for.

    Returns:
    int: New user ID assigned for the added ratings.
    """
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("data/ratings.csv", index=False)
        return new_id
