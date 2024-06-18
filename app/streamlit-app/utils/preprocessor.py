import numpy as np
import pandas as pd
from utils.data_loader import load_ratings, load_course_genres

def get_user_vector(user):
    """
    Generates a user profile vector based on their ratings and course genres.

    Args:
    user (int): User ID for whom the profile vector is generated.

    Returns:
    DataFrame: Pandas DataFrame containing the user profile vector with genre weights.
               Columns represent different course genres.

    Notes:
    - The function loads ratings and course genres data using `load_ratings()` and `load_course_genres()` functions.
    - Calculates a user profile vector `u0` where each element represents the user's rating for a specific course.
    - Multiplies `u0` by the course genre matrix `C` to compute genre weights.
    - Returns a DataFrame `user_profile_df` where rows correspond to the user's genre weights.
    """
    ratings_df = load_ratings()
    course_genres_df = load_course_genres()
    all_courses = set(course_genres_df['COURSE_ID'].values)
    u_weights = []
    user_courses = ratings_df[ratings_df['user']==user]['item'].unique()
    all_courses = course_genres_df['COURSE_ID'].unique()
    u0 = np.zeros((1, len(all_courses)), dtype='int32')
    for i, course in enumerate(all_courses):
        if course in user_courses:
            y = ratings_df[(ratings_df['user'] == user) & (ratings_df['item'] == course)]['rating'].values[0]
            u0[0, i] = y
        else:
            u0[0, i] = 0
    genres = [x for x in course_genres_df.columns if x not in ['COURSE_ID', 'TITLE']]
    C = course_genres_df[genres].to_numpy()
    #print(f"User profile vector shape {u0.shape} and course genre matrix shape {C.shape}")
    u0_weights = np.matmul(u0, C)
    #print(user, u0_weights)
    u_weights.append(u0_weights.reshape(1, len(genres)))

    u_weights = tuple(u_weights)

    weights = np.concatenate(u_weights, axis=0)
    user_profile_df = pd.DataFrame(weights, columns=genres)
    return  user_profile_df 