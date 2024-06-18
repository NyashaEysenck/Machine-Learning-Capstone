"""
This module provides functions for training a clustering model using K-means algorithm 
and generating course recommendations based on user clusters.

Imports:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from joblib import dump
    import numpy as np
    import pandas as pd
    from utils.preprocessor import get_user_vector
    from utils.data_loader import load_ratings, load_user_profiles, models

Global Variables:
    RS (int): Random state used for reproducibility.

Functions:
    train_clustering(user_profile_df, ratings_df, model_name, cluster_no, component_no, pca=False):
        Trains a K-means clustering model on user profiles and returns course clusters.

    clustering_recommendations(user, cluster_no=14, component_no=9, pca=False):
        Generates course recommendations for a user based on clustering results.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from utils.preprocessor import get_user_vector
from utils.data_loader import load_ratings, load_user_profiles, models
 
RS = 12

def train_clustering(user_profile_df,ratings_df, model_name, cluster_no, component_no, pca=False):
    """
    Trains a K-means clustering model on user profiles and returns course clusters.

    Args:
        user_profile_df (pandas.DataFrame): DataFrame containing user profiles with features.
        ratings_df (pandas.DataFrame): DataFrame containing user-course ratings.
        model_name (str): Name of the model.
        cluster_no (int): Number of clusters to generate.
        component_no (int): Number of principal components for PCA.
        pca (bool, optional): Whether to use PCA for dimensionality reduction. Defaults to False.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: DataFrame with grouped courses and enrollments per cluster.
            - pandas.DataFrame: DataFrame with labeled clusters for users.
    """
    def combine_cluster_labels(user_ids, labels):
        """
        Combines user IDs with cluster labels.

        Args:
            user_ids (pandas.DataFrame): DataFrame containing user IDs.
            labels (numpy.ndarray): Array of cluster labels.

        Returns:
            pandas.DataFrame: DataFrame with 'user' and 'cluster' columns.
        """
        # Convert labels to a DataFrame
        labels_df = pd.DataFrame(labels)    
        # Merge user_ids DataFrame with labels DataFrame based on index
        cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
        # Rename columns to 'user' and 'cluster'
        cluster_df.columns = ['user', 'cluster']
        return cluster_df

    print("\nTraining: " + model_name + "\n")

    feature_names = list(user_profile_df.columns[1:])

    # Use StandardScaler to make each feature with mean 0, standard deviation 1
    # Instantiating a StandardScaler object
    scaler = StandardScaler()
    # Standardizing the selected features (feature_names) in the user_profile_df DataFrame
    user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
    features = user_profile_df.loc[:, user_profile_df.columns != 'user']
    user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
    cluster_labels = [None] * len(user_ids)

    if pca:
        pca_model = PCA(n_components=component_no, random_state=RS).fit(features)
        components = pca_model.transform(features)
        components_df = pd.DataFrame(data=components)
        transformed_df = pd.merge(user_ids, components_df, left_index=True, right_index=True)
        transformed_features =transformed_df.loc[:, transformed_df.columns != 'user']
        model = KMeans(n_clusters=cluster_no, random_state=RS).fit(transformed_features)
    else:
        model = KMeans(n_clusters=cluster_no, random_state=RS).fit(features)
 
    cluster_labels = model.labels_
    cluster_df = combine_cluster_labels(user_ids, cluster_labels)
    clusters_labelled = pd.merge(ratings_df, cluster_df, left_on='user', right_on='user')
    # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
    courses_cluster = clusters_labelled[['item', 'cluster']]
    # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
    courses_cluster['count'] = [1] * len(courses_cluster)
    # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
    # and resetting the index to make the result more readable
    courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()
    return courses_cluster_grouped, clusters_labelled

def clustering_recommendations(user, cluster_no=14, component_no=9, pca=False):
    """
    Generates course recommendations for a user based on clustering results.

    Args:
        user (int): User ID for whom recommendations are generated.
        cluster_no (int, optional): Number of clusters used in clustering. Defaults to 14.
        component_no (int, optional): Number of PCA components. Defaults to 9.
        pca (bool, optional): Whether to use PCA for clustering. Defaults to False.

    Returns:
        tuple: Tuple containing lists of:
            - int: User IDs.
            - int: Recommended course IDs.
            - float: Recommendation scores.
    """
    user_profile_df = load_user_profiles()
    ratings_df = load_ratings()
    features = get_user_vector(user)
    # Create a new row with the user and the array
    new_user = np.insert(features, 0, user, axis=1)
    # Convert the new row to a DataFrame
    new_user_df = pd.DataFrame(new_user, columns=user_profile_df.columns)
 
    # Append the new row to the existing DataFrame
    df = pd.concat([new_user_df , user_profile_df], ignore_index=True)
    courses_cluster_grouped, clusters_labelled = train_clustering(df, ratings_df, models[2], cluster_no, component_no, pca=pca)
 
    cluster_label = clusters_labelled[clusters_labelled['user'] == user]['cluster'].values[0]
 
    courses_cluster_grouped = courses_cluster_grouped.copy()[courses_cluster_grouped['cluster'] == cluster_label]
 
    courses_cluster_grouped.sort_values(by='enrollments', ascending=False, inplace=True)
 
    user_subset = ratings_df[ratings_df['user'] == user]

    enrolled_courses = user_subset['item'].tolist()

    new_courses = []
    users=[]
    scores=[]
 
    for _, course, score in courses_cluster_grouped.values:
        if course in enrolled_courses:
            continue
        new_courses.append(course)
        users.append(user)
        scores.append(score)
    
    return users, new_courses, scores
