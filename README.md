# Disaster Response Pipeline Project

### Project Summary:
In this project, I applied the skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

Using the datasets, I created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 



### Contents:
- README.md
- \app:
    - run.py: script that runs the flask app
- \data:
    - disaster_categories.csv: categories file
	- disaster_messages.csv: messages file
	- DisasterResponse.db: database
	- process_data.py: ETL script
- \models:
    train_classifier.py: classification model script
    classifier.pkl: classifier pickle file (too large for GitHub)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Resources
https://gist.github.com/gruber/8891611
https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
