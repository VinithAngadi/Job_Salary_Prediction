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
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import io
import csv
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

nltk.download('stopwords')


def load_data(data_filepath):

    """
    Read csv file of scraped data from indeed.com containing job title, job 
    description, location, cost index, purchasing power index of each state,
    and the target variable, Salaries
    
    Input: data_filepath: path of scraped data('cleaned_jobs.csv')

    Output: df: dataframe object of the scraped data
    """

    # read data from csv file
    df = pd.read_csv(data_filepath)
    
    return df

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

def NMF_cluster(data, min_n, max_n, vocab):

    n_samples = 20000
    NMF_topics = 10
    NMF_top_words = 10
    n_features = len(vocab)

    # Use tf-idf features for NMF.
    print("\nExtracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1,
                                       vocabulary=vocab,
                                       stop_words='english',
                                       token_pattern=r"(?u)\S+",
                                       max_features=n_features, sublinear_tf=True,
                                       ngram_range=(min_n, max_n))

    t0 = pd.datetime.now()
    tfidf = tfidf_vectorizer.fit_transform(data)
    print("done in: " ,(pd.datetime.now() - t0))

    # Fit the NMF model
    print("\nFitting the NMF model with tf-idf features"
          "\nn_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = pd.datetime.now()
    nmf = NMF(n_components=NMF_topics, init='nndsvda',
              random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in: " ,(pd.datetime.now() - t0))

    print("\nNMF model completed, now generating topic labels\n")

    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    tfidf_skills_df = pd.DataFrame(tfidf_matrix.todense(),columns=tfidf_vectorizer.get_feature_names())
    topic_values = nmf.transform(tfidf_matrix)
    tfidf_skills_df['topic'] = topic_values.argmax(axis=1)

    return tfidf_skills_df, tfidf_vectorizer, nmf

def build_model():
#     model = RandomForestRegressor(max_features = 1400, max_depth = 60, n_estimators = 450, random_state = 0)
    model = ensemble.GradientBoostingRegressor(learning_rate = 0.1, max_depth = 10, n_estimators = 1000, random_state = 0)


    return model


def mean_absolute_percentage_error(y_test, y_pred):
    val = 0
    for t, p in zip(y_test, y_pred):
        val += abs((t - p)/t)
    return (val/len(y_test))

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    Y_pred = model.predict(X_test)
    model_rf_r2 = model.score(X_train, Y_train) 
    model_rf_mae = mean_absolute_error(Y_test,Y_pred)
    model_rf_mape = mean_absolute_percentage_error(Y_test, Y_pred)
    print('R^2 : ', model_rf_r2)
    print('Test MAPE', model_rf_mape)
    print('Test MAE : ',model_rf_mae)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 6:
        data_filepath, model_filepath, vocab_filepath, tfidf_filepath, NMF_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(data_filepath))
        df = load_data(data_filepath)
        df_temp = df.drop(['Unnamed: 0','state','salary_offered'], axis = 1)

        X = df_temp
        Y = df['salary_offered']

        print('Cleaning job description...')
        X = clean_job_description(X)
        print(X.columns)

        print('Extracting skills from the vocabulary...')
        skills = obtain_skills(vocab_filepath)

        print('Generating TF-IDF matrix and executing topic modelling...')
        tfidf_skills_df, tfidf_vectorizer, nmf = NMF_cluster(X['job_description_lower'], 1, 3, skills)

        X = pd.concat([X,tfidf_skills_df],axis=1)
        # X = X.fillna(0)
        X = X.drop(['job_description','job_description_lower','topic'],axis=1)

        scale = StandardScaler().fit(X)
        X = pd.DataFrame(scale.transform(X), columns = X.columns)
        X = pd.concat([X, df['state'],tfidf_skills_df['topic']], axis = 1)
        X = pd.get_dummies(X, columns=['state', 'topic'])
        X = X.fillna(0)
        print(X.columns.values.tolist())

        # split into train and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_train, Y_train, X_test, Y_test)

        print('Saving tfidf vectorizer...\n    MODEL: {}'.format(tfidf_filepath))
        save_model(tfidf_vectorizer, tfidf_filepath)

        print('Saving NMF model...\n    MODEL: {}'.format(NMF_filepath))
        save_model(nmf, NMF_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the job description data '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument.')


if __name__ == '__main__':
    main()
