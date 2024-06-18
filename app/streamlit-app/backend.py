"""
This module provides functions for training and predicting course recommendations using
various content-based models. It includes functionality to train models based on course
similarity, clustering, and user profile analysis, as well as generating recommendations
for users based on these models.

Imports:
    pandas as pd
    numpy as np
    from models.content_based.course_similarity import (
        train_course_similarity,
        course_similarity_recommendations,
        get_doc_dicts
    )
    from models.content_based.clustering import clustering_recommendations
    from models.content_based.user_profile import user_profile_recommendations
    from utils.data_loader import *

Functions:
    train(model_name, params):
        Trains a specified model using provided parameters.

    predict(model_name, user_ids, params):
        Generates course recommendations for given users based on a specified model.
"""

import pandas as pd
from models.content_based.course_similarity import (
    train_course_similarity,
    course_similarity_recommendations,
    get_doc_dicts
)
from models.content_based.clustering import clustering_recommendations
from models.content_based.user_profile import user_profile_recommendations
from utils.data_loader import models, load_courses, load_bow, load_course_sims, load_ratings, add_new_ratings

# Model training
def train(model_name, params):
    """
    Trains a specified model using provided parameters.

    Args:
        model_name (str): The name of the model to train. Supported models include
                          course similarity, user profile, and clustering-based models.
        params (dict): A dictionary of parameters required for training the model.

    Returns:
        None
    """
    courses_df = load_courses()
    bows_df = load_bow()
    if model_name == models[0]:
        train_course_similarity(courses_df.copy(), bows_df.copy(), model_name)

# Prediction
def predict(model_name, user_ids, params):
    """
    Generates course recommendations for given users based on a specified model.

    Args:
        model_name (str): The name of the model to use for generating recommendations.
                          Supported models include course similarity, user profile,
                          and clustering-based models.
        user_ids (list): A list of user IDs for whom to generate recommendations.
        params (dict): A dictionary of parameters for generating recommendations, such as
                       the number of top courses to return and various model-specific thresholds.

    Returns:
        pandas.DataFrame: A DataFrame containing the recommended courses for each user,
                          along with the recommendation scores.
    """
    top_courses = 10

    if "top_courses" in params:
        top_courses = params["top_courses"]

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            if "course_sim_threshold" in params:
                sim_threshold = params["course_sim_threshold"] / 100.0
            else:
                sim_threshold = 0.6

            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict,
                                                    id_idx_dict,
                                                    enrolled_course_ids,
                                                    sim_matrix
                                                    )
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        if model_name == models[1]:

            if "profile_sim_threshold" in params:
                sim_threshold = params["profile_sim_threshold"]
            else:
                sim_threshold = 14

            users, courses, scores = user_profile_recommendations(user_id, sim_threshold)

        if model_name == models[2]:
            if "number_of_clusters" in params:
                cluster_no = params["number_of_clusters"]
            else:
                cluster_no = 14

            users, courses, scores = clustering_recommendations(user_id, cluster_no)

        if model_name == models[3]:
            if "number_of_clusters" in params:
                cluster_no = params["number_of_clusters"]
            else:
                cluster_no = 14

            if "number_of_components" in params:
                component_no = params["number_of_components"]
            else:
                component_no = 9

            users, courses, scores = clustering_recommendations(user_id,
                                                                cluster_no,
                                                                component_no,
                                                                pca=True
                                                                )

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df.iloc[:top_courses]
