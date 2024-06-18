# Course-Recommender-System
IBM Machine Learning Capstone Project

This project was developed as part of the IBM Machine Learning Certificate. It includes data analysis, content filtering and collaborative filtering modelling, and a Streamlit application.

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for data analysis and model building.
- `app/`: Contains the Streamlit application.
- `reports/`: Contains the presentation report.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Course-Recommender-System.git
2. Navigate to the app directory and install the required packages:
    cd app
    pip install -r requirements.txt

### Usage

To run the Streamlit app:
streamlit run recommender_app.py

### Retraining Models
- **Course Similarity Model:** All models are pretrained, but you can retrain the `course_similarity` model with new data. 
  
- **Clustering Model:** The clustering model retrains every time you make a prediction. No additional action is required to retrain this model.

### Updating Datasets
- Replace the datasets in the `data` folder of the Streamlit app (`app/data/`) with your own datasets if need be.

### License
This project is licensed under the MIT License.