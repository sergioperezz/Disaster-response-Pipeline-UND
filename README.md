# Disaster response Pipeline Project

## Project motivation

The project goes from the ETL of the data, creating a model to classify the messages and a web to plot the result. The project gets a message and classifies it.

## Content
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app


- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- Database.db   # database to save clean data to
|- ETL Pipeline Preparation.ipynb

- models
|- train_classifier.py
|- classifier.pkl  # saved model 
|- ML pipeline prepatation.ipynb

- README.md
```

## Installation

### Dependencies
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

### User installation
Clone the repository:
```
https://github.com/sergioperezz/Disaster-response-Pipeline-UND.git
```
## Running the tests 

```
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
```

## Authors

Sergio Pérez

## License

## Capture

![alt text](https://github.com/sergioperezz/Disaster-response-Pipeline-UND/blob/master/captures/front1.PNG)
![alt text](https://github.com/sergioperezz/Disaster-response-Pipeline-UND/blob/master/captures/front2.PNG)
![alt text](https://github.com/sergioperezz/Disaster-response-Pipeline-UND/blob/master/captures/front3.PNG)
