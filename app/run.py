import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import sys
import pandas as pd
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import io
import csv
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

app = Flask(__name__)

# load data
df = pd.read_csv("data/Cleaned_final.csv")

# load model
model = joblib.load("models/predictor.pkl")
tfidf_vect = joblib.load("models/tfidf.pkl")
nmf = joblib.load("models/nmf.pkl")
vocab_filepath = "data/vocab.txt"


def clean_job_description(df):

    # Convert to lower case
    df['job_description_lower'] = [x.lower() for x in df['job_description']]

    # Remove punctuations
    df['job_description_lower'] = [re.sub(r"[^a-zA-Z0-9]", " ",x) for x in df['job_description_lower']]

    return df

def obtain_skills(vocab_filepath):

    # import stop words set from NLTK package
    stop_word_set = set(stopwords.words('english'))

    # read text file of vocabulary of skills to extract the technical, non-technical,
    # and managerial skills

    with io.open(vocab_filepath) as source:
        rdr = csv.reader(source)
        skills = []
        skill_set = set()
        for row in rdr:
            skill = row[0].lower()
            if skill not in skill_set and skill not in stop_word_set:
                skill_set.add(skill)
                skills.append(skill)
    return skills


def get_cost_value_index(state):
  df_cost_index_value = pd.read_csv('data/Cost_index_value.csv')
  return df_cost_index_value[df_cost_index_value['state'] == state]['value'].values[0]\
  , df_cost_index_value[df_cost_index_value['state'] == state]['costIndex'].values[0]

def generate_one_hot_encode(state, topic_number, tfidf_skills_df):
  states = ['FL', 'MI', 'ID', 'MO', 'CA', 'WA', 'MA', 'UT', 'VA', 'OR', 'TX',
       'OH', 'PA', 'HI', 'CO', 'MD', 'WI', 'AZ', 'GA', 'NY', 'NH', 'NJ',
       'R', 'PR', 'TN', 'WV', 'IL', 'VT', 'AL', 'IN', 'OK', 'MS', 'NC',
       'NV', 'SC', 'NE', 'KY', 'RI', 'MN', 'LA', 'WY', 'SD', 'MT', 'KS',
       'NM', 'AR', 'IA', 'CT', 'ME', 'AK', 'ND', 'DE']
  states = sorted(states)
  topic_number = str(topic_number)
  topics = ['0','1','2','3','4','5','6','7','8','9']
  dict_df = {}
  dict_df_2 = {}

  cost_values = get_cost_value_index(state)
  dict_df['cost_index'] = cost_values[1]
  dict_df['value'] = cost_values[0]

  df_1 = pd.DataFrame(dict_df,index=[0])
  df_2 = pd.concat([df_1,tfidf_skills_df],axis=1)

  if state in states:
    for s in states:
      if s != state:
        dict_df_2['state_'+s] = 0.0
      else:
        dict_df_2['state_'+s] = 1.0
        
  if topic_number in topics:
    for t in topics:
      if t != topic_number:
        dict_df_2['topic_'+t] = 0.0
      else:
        dict_df_2['topic_'+t] = 1.0
  df_3 = pd.DataFrame(dict_df_2, index=[0])

  df_res = pd.concat([df_2,df_3],axis=1)
  return df_res


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    mean_salary = df.groupby('topic').mean()['salary_offered']
    mean_salary_state = df.groupby('state').mean()['salary_offered']
    topics = list(mean_salary.index)
    states = list(mean_salary_state.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=topics,
                    y=mean_salary
                )
            ],

            'layout': {
                'title': 'Distribution of Salaries',
                'yaxis': {
                    'title': "Mean Salary"
                },
                'xaxis': {
                    'title': "Job Title"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=states,
                    y=mean_salary_state
                )
            ],

            'layout': {
                'title': 'Distribution of Salaries by State',
                'yaxis': {
                    'title': "Mean Salary"
                },
                'xaxis': {
                    'title': "State"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    state = request.args.get('state','') 
    query = [query]
    state = [state]

    df_state = pd.DataFrame(state,columns=['state'])
    df = pd.DataFrame(query,columns=['job_description'])
    df_clean = clean_job_description(df)

    skills = obtain_skills(vocab_filepath)

    tfidf_matrix = tfidf_vect.transform(df_clean['job_description_lower'])
    tfidf_skills_df = pd.DataFrame(tfidf_matrix.todense(),columns=tfidf_vect.get_feature_names())
    topic_values = nmf.transform(tfidf_matrix)
    tfidf_skills_df['topic'] = topic_values.argmax(axis=1)
    df_query = generate_one_hot_encode(df_state['state'].iloc[0],tfidf_skills_df['topic'].iloc[0]\
        ,tfidf_skills_df.iloc[:,:len(tfidf_skills_df.columns)-1])

    # use model to predict classification for query
    salary_pred = round(model.predict(df_query)[0],3)
 
    return render_template(
        'go.html',
        query=query,
        state=state,
        salary=salary_pred
    )


def main():
    app.run(host='127.0.0.1', port=8080, debug=True)


if __name__ == '__main__':
    main()