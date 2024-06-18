"""
This module provides functions for training a course similarity model based on bag-of-words (BoW) representations
and generating course recommendations using cosine similarity.

Imports:
    from scipy.spatial.distance import cosine
    import pandas as pd
    import numpy as np
    from utils.data_loader import load_bow

Global Variables:
    RS (int): Random state used for reproducibility.

Functions:
    train_course_similarity(courses_df, bows_df, model_name):
        Trains a course similarity model and saves the similarity matrix to a CSV file.

    get_doc_dicts():
        Retrieves mappings between document indices and document IDs from the loaded BoW DataFrame.

    course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
        Generates course recommendations based on user-enrolled courses and a precomputed similarity matrix.
"""
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
from utils.data_loader import load_bow

RS = 12

def train_course_similarity(courses_df, bows_df, model_name):
    """
    Trains a course similarity model and saves the similarity matrix to a CSV file.

    Args:
        courses_df (pandas.DataFrame): DataFrame containing course information.
        bows_df (pandas.DataFrame): DataFrame containing bag-of-words representations for courses.
        model_name (str): Name of the model.

    Saves:
        CSV file: Similarity matrix saved to 'data/sim.csv'.
    """
    def pivot_two_bows(basedoc, comparedoc):
        """
        Pivot two bag-of-words (BoW) representations for comparison.

        Parameters:
        basedoc (DataFrame): DataFrame containing the bag-of-words representation for the base document.
        comparedoc (DataFrame): DataFrame containing the bag-of-words representation for the document to compare.

        Returns:
        DataFrame: A DataFrame with pivoted BoW representations for the base and compared documents,
        facilitating direct comparison of word occurrences between the two documents.
        """
        # Create copies of the input DataFrames to avoid modifying the originals
        base = basedoc.copy()
        base['type'] = 'base'  # Add a 'type' column indicating base document
        compare = comparedoc.copy()
        compare['type'] = 'compare'  # Add a 'type' column indicating compared document

        # Concatenate the two DataFrames vertically
        join = pd.concat([base, compare])

        # Pivot the concatenated DataFrame based on 'doc_id' and 'type', with words as columns
        joinT = join.pivot(index=['doc_id', 'type'], columns='token').fillna(0).reset_index(level=[0, 1])

        # Assign meaningful column names to the pivoted DataFrame
        joinT.columns = ['doc_id', 'type'] + [t[1] for t in joinT.columns][2:]

        # Return the pivoted DataFrame for comparison
        return joinT
        
    print("\nTraining: " + model_name + "\n")
    course_ids = courses_df.index
    similarity_matrix = pd.DataFrame(np.zeros((len(course_ids), len(course_ids))), index=course_ids, columns=course_ids)
    course_list = [x for x in bows_df['doc_id'].unique().tolist()]

    for first_course in course_list:
        # Step 1: Retrieve the BoW feature vector for the course ML0101ENv3
        ml_bow = bows_df[bows_df['doc_id'] == first_course]
        for other_course in course_list:
            other_bow =  bows_df[bows_df['doc_id'] ==other_course]
            pivoted_bows = pivot_two_bows(ml_bow, other_bow)
            similarity = 1 - cosine(pivoted_bows.iloc[0, 2:], pivoted_bows.iloc[1, 2:])
            i = courses_df[courses_df['COURSE_ID']==first_course].index[0]
            j = courses_df[courses_df['COURSE_ID']==other_course].index[0]
            similarity_matrix.iloc[i, j] = similarity

    #Saving the matrix to a CSV file
    csv_file = 'data/sim.csv'
    similarity_matrix.to_csv(csv_file)
    print(f"Similarity matrix saved to {csv_file}")
 
# Create course id to index and index to id mappings
def get_doc_dicts():
    """
    Retrieves mappings between document indices and document IDs from the loaded BoW DataFrame.

    Returns:
        tuple: Tuple containing dictionaries:
            - dict: Mapping from index to document ID.
            - dict: Mapping from document ID to index.
    """
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    """
    Generates course recommendations based on user-enrolled courses and a precomputed similarity matrix.

    Args:
        idx_id_dict (dict): Mapping from index to course ID.
        id_idx_dict (dict): Mapping from course ID to index.
        enrolled_course_ids (list): List of course IDs that the user has already enrolled in.
        sim_matrix (numpy.ndarray): Precomputed similarity matrix between courses.

    Returns:
        dict: Dictionary containing recommended course IDs as keys and similarity scores as values,
        sorted in descending order of similarity score.
    """
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res
