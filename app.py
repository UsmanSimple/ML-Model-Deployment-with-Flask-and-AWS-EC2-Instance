from flask import Flask
from flask import render_template, request, url_for, jsonify
import pickle
import pandas
import os
import numpy 

STATIC_DIR = os.path.abspath('./static')
app = Flask(__name__,static_folder=STATIC_DIR)

app = Flask(__name__) #initialize the flask app

model = pickle.load(open('models/pipeline_LGBM.pkl','rb')) # load the data via pickle

column =['country', 'year', 'status', 'adult_mortality', 'infant_deaths',
       'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles', 'bmi',
       'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria',
       'hiv/aids', 'gdp', 'thinness__1-19_years', 'thinness_5-9_years',
       'income_composition_of_resources', 'schooling', 'region', 'incomegroup',
       'Population']

@app.route("/")
def index():
    return render_template("Home.html")


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
    percentage_expenditure = float(req.get("percentage_expenditure",default=2.3))
    hepatitis_b = int(req.get("hepatitis_b",default=88))
    measles = float(req.get("measles",default=6708))
    bmi = float(req.get("bmi",default=25))
    under_five_deaths = int(req.get("under_five_deaths",default=513))
    polio = float(req.get("polio",default=80))
    total_expenditure = float(req.get("total_expenditure",default=7))
    diphtheria = float(req.get("diphtheria",default=24))
    hiv_aids = float(req.get("hiv_aids",default=100))
    gdp = float(req.get("gdp",default=483000))
    thinness_10_19_years = float(req.get("thinness_10_19_years",default=13.2))
    thinness_5_9_years = float(req.get("thinness_5_9_years",default=13.2))
    income_composition_of_resources = float(req.get("income_composition_of_resources",default=0.52))
    schooling = float(req.get("schooling",default=10.9))
    region = req.get("region",default='Sub-Saharan Africa')
    incomegroup = req.get("incomegroup",default='Lower middle income')
    population = float(req.get("population",default=206140000))

        #storing in array
    array = numpy.array([country,year,status,adult_mortality,infant_deaths,
                            alcohol,percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure,
                            diphtheria, hiv_aids, gdp, thinness_10_19_years, thinness_5_9_years, income_composition_of_resources,  
                                 schooling, region, incomegroup, population]
                                ).reshape(1,23)

    #creates a dataframe to hold the data and perform transformation
    data = pandas.DataFrame(data=array,columns=['country', 'year', 'status', 'adult_mortality', 'infant_deaths',
       'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles', 'bmi',
       'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria',
       'hiv/aids', 'gdp', "thinness__1-19_years", 'thinness_5-9_years',
       'income_composition_of_resources', 'schooling', 'region', 'incomegroup',
       'Population'])

    #predict over the features gotten from the user
    prediction = model.predict(data)
    output = round(prediction[0])

        #passing value gotten to template for rendering
    return render_template("Predict.html",prediction_text='The Average Life Expectancy of {} in year {} is {} years'.format(country, year, output))    

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


if __name__ == "__main__":
    app.run(debug=True)