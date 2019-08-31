# Disaster Response Pipeline Project

## Table of Contents
1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Required Libraries](#libraries)
4. [Instructions](#instructions)


## Project Motivation <a name="motivation"></a>

This project focuses on applying data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The model is used to categorize disaster events, so messages can be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## File Descriptions <a name="files"></a>

There are three main files, that are needed to complete for this project:

### **1. process_data.py**, which Includes data cleaning pipeline that

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
 ### **2.train_classifier.py**, which includes machine learning pipeline that
 
 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
 ### **3.run.py**, web app that uses the trained model to input text and return classification results.

## Required Libraries <a name="libraries"></a>

 - Machine Learning Libraries: Pandas, NumPy, Sciki-Learn
 - Natural Language Process Libraries: NLTK
 - SQLlite Database Libraqries: SQLalchemy
 - Web App libraries: Flask
 - Visualization libraries: Plotly

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
