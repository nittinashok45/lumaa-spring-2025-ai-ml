# Content-Based Movie Recommendation System

A simple content-based recommendation system that suggests movies based on a short text description of user preferences.

## Overview

- **What It Does:**  
  Converts movie descriptions into TF-IDF vectors and computes cosine similarity with a user query to output the top movie recommendations.

- **Dataset:**  
  Use your own sample of a movie dataset (e.g., a subset of the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) or [IMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)). The example in `code.py` uses a small in-code CSV.

## Requirements

- **Python:** 3.7+
- **Dependencies:** Listed in `requirements.txt` (e.g., pandas, scikit-learn).

## Run
**pip install -r requirements.txt**
**python code.py**
