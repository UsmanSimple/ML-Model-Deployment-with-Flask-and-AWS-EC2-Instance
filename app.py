# import required libraries
from flask import Flask
from flask import render_template, request, url_for, jsonify
import pickle
import pandas
import os
import numpy 

# Assign parameters such as static folder names- static

STATIC_DIR = os.path.abspath('./static')

#initialize the flask app
app = Flask(__name__,static_folder=STATIC_DIR)

# load the data via pickle file for prediction
model = pickle.load(open('models/pipeline_LGBM.pkl','rb')) 

# state the columns for inputing variables
column =['country', 'year', 'status', 'adult_mortality', 'infant_deaths',
       'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles', 'bmi',
       'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria',
       'hiv/aids', 'gdp', 'thinness__1-19_years', 'thinness_5-9_years',
       'income_composition_of_resources', 'schooling', 'region', 'incomegroup',
       'Population']

# rendering the homepage template with index function
@app.route("/")
def index():
    return render_template("Home.html")

# rendering the predict.html as it contains the parameters needed for the prediction.
# it requires both the GET for the default value prediction and POST request for prediction if pred.html forms is filled by the User 
@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    #creates the single function for getting data
    
    req = request.form
    country = req.get("country",default='Nigeria')
    year = int(req.get("year",default=2021))
    status = req.get("status",default= 'Developing')
    adult_mortality = int(req.get("adult_mortality",default=343))
    infant_deaths = int(req.get("infant_deaths",default=72))
    alcohol = float(req.get("alcohol",default=13))
    percentage_expenditure = float(req.get("percentage_expenditure",default=2300))
    hepatitis_b = float(req.get("hepatitis_b",default=88))
    measles = float(req.get("measles",default=6718))
    bmi = float(req.get("bmi",default=25))
    under_five_deaths = int(req.get("under_five_deaths",default=114))
    polio = float(req.get("polio",default=80))
    total_expenditure = float(req.get("total_expenditure",default=7))
    diphtheria = float(req.get("diphtheria",default=24))
    hiv_aids = float(req.get("hiv_aids",default=100))
    gdp = float(req.get("gdp",default=480000))
    thinness_10_19_years = float(req.get("thinness_10_19_years",default=13.4))
    thinness_5_9_years = float(req.get("thinness_5_9_years",default=13.2))
    income_composition_of_resources = float(req.get("income_composition_of_resources",default=0.52))
    schooling = float(req.get("schooling",default=10.9))
    region = req.get("region",default='Sub-Saharan Africa')
    incomegroup = req.get("incomegroup",default='Lower middle income')
    population = float(req.get("population",default=214140000))

    #storing the data in array format
    array = numpy.array([country,year,status,adult_mortality,infant_deaths,
                            alcohol,percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure,
                            diphtheria, hiv_aids, gdp, thinness_10_19_years, thinness_5_9_years, income_composition_of_resources,  
                                 schooling, region, incomegroup, population]
                                ).reshape(1,23)

    #creates a dataframe to hold the data
    data = pandas.DataFrame(data=array,columns=['country', 'year', 'status', 'adult_mortality', 'infant_deaths',
       'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles', 'bmi',
       'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria',
       'hiv/aids', 'gdp', "thinness__1-19_years", 'thinness_5-9_years',
       'income_composition_of_resources', 'schooling', 'region', 'incomegroup',
       'Population'])

    #predict over the features gotten from the user with the model
    prediction = model.predict(data)
    output = round(prediction[0])

    #passing value gotten to Predict html template for rendering
    return render_template("Predict.html",prediction_text='The Average life expectancy for {} in year {} is {} years'.format(country, year, output))    

# send direct POST requests to Flask API using the command python request.py
@app.route('/predict_api',methods=['POST'])
def results():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    input = pandas.DataFrame(data.values(),columns=column)

    prediction = model.predict(data)
    output = round(prediction[0])

    return jsonify(output)

# port values changes as we run on local server and in the cloud

if __name__ == '__main__':
   app.run(host = '0.0.0.0', port = 8080) # cloud

#if __name__ == "__main__":
# app.run(debug=True) # local machine port= 5000, host address= 127.0.0.1