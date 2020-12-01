import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

df_tfidf = pd.read_csv('tfidf_skills_df.csv')
df_topics = pd.read_csv('df_topic_updated.csv')
df_cleaned_jobs = pd.read_csv('Cleaned_jobs.csv')

df_tfidf.drop('Unnamed: 0', axis = 1, inplace = True)

df_cleaned_jobs.drop(['Unnamed: 0','job_title', 'job_description','job_type'], axis = 1, inplace = True)

df_cost = pd.read_csv('cost_index.csv')
df_cost = df_cost[['costIndex', 'Abb', 'Real Value of $100']]
df_cost.rename(columns = {'Abb':'state', 'Real Value of $100' : 'value'}, inplace = True)
df_temp = pd.DataFrame({'costIndex' : [100.0, 100.0], 'state':['PR', 'R'], 'value':['$100.00', '$100.00']})
df_cost = pd.concat([df_cost, df_temp], ignore_index = True)
val_list = []
for i,rows in df_cost.iterrows():
    val = rows['value']
    val_list.append(float(val[1:]))
df_cost['value'] = val_list

df_cleaned_jobs = pd.merge(df_cleaned_jobs, df_cost, on='state', how='left')

topics_list = list(df_topics['Topic'])


t_list = []
for i in topics_list:
    if i == 0:
        t_list.append('Data Science')
    elif i == 1:
        t_list.append('Manager')
    elif i == 2:
        t_list.append('Sales and Marketing')
    elif i == 3:
        t_list.append('Engineering Manager')
    elif i == 4:
        t_list.append('Customer Service')
    elif i == 5:
        t_list.append('Mechanical Engineer')
    elif i == 6:
        t_list.append('Data Analytics')
    elif i == 7:
        t_list.append('Accounting')
    elif i == 8:
        t_list.append('Software Engineer')
    elif i == 9:
        t_list.append('Cashier')


df_cleaned_jobs['topic'] = t_list

X_without = df_cleaned_jobs[['state','costIndex','value','topic']]
y_without = df_cleaned_jobs['salary_offered']

X_without = pd.get_dummies(X_without, columns=['state', 'topic'])

X = pd.concat([X_without, df_tfidf], axis = 1)
y = y_without

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)
X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without,y_without, test_size = 0.2, random_state = 1)


cols = list(df_tfidf.columns)
cols.extend(['value','costIndex'])

scale = StandardScaler().fit(X_train[cols])

X_train[cols] = scale.transform(X_train[cols])
X_test[cols] = scale.transform(X_test[cols])

from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_test, y_pred):
    val = 0
    for t, p in zip(y_test, y_pred):
        val += abs((t - p)/t)
    return (val/len(y_test))

# Linear Regression
# with tfidf features

from sklearn.linear_model import LinearRegression

model_lr = LinearRegression().fit(X_train, y_train)
model_lr_r2 = model_lr.score(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
model_lr_mae = mean_absolute_error(y_test, y_pred_lr)
model_lr_mape = mean_absolute_percentage_error(y_test,y_pred_lr)

print('R^2:',model_lr_r2)
print('Test MAE:',model_lr_mae)
print('Test MAPE:', model_lr_mape)
print('Coefficients:',model_lr.coef_)

# without tfidf

model_lr_without = LinearRegression().fit(X_train_without, y_train_without)
model_lr_r2_without = model_lr_without.score(X_train_without, y_train_without)
y_pred_lr_without = model_lr_without.predict(X_test_without)
model_lr_mae_without = mean_absolute_error(y_test_without, y_pred_lr_without)
model_lr_mape_without = mean_absolute_percentage_error(y_test_without,y_pred_lr_without)

print('R^2:',model_lr_r2_without)
print('Test MAE:',model_lr_mae_without)
print('Test MAPE:', model_lr_mape_without)
print('Coefficients:',model_lr_without.coef_)


# Ridge Ridge Regression
# with tfidf

alphas = np.linspace(0.001,1,100)
clf_rr = RidgeCV(alphas=alphas).fit(X_train, y_train)

model_rr = Ridge(alpha = clf_rr.alpha_).fit(X_train,y_train)
model_rr_r2 = model_rr.score(X_train, y_train)
y_pred_rr = model_rr.predict(X_test)
model_rr_mae = mean_absolute_error(y_test, y_pred_rr)

print('R^2 : ',model_rr_r2)
print('Test MAE : ',model_rr_mae)
print('Test MAPE:', mean_absolute_percentage_error(y_test,y_pred_rr))
print('Coeffficients:', model_rr.coef_)
print('Intercept:', model_rr.intercept_)

# without tfidf

clf_without_rr = RidgeCV(alphas=alphas).fit(X_train_without, y_train_without)
clf_without_rr.alpha_

model_rr_without = Ridge(alpha = clf_without_rr.alpha_).fit(X_train_without, y_train_without)
model_rr_r2_without = model_rr_without.score(X_train_without, y_train_without)
y_pred_rr_without = model_rr_without.predict(X_test_without)
model_rr_mae_without = mean_absolute_error(y_test_without, y_pred_rr_without)
print('Test R^2 : ',model_rr_r2_without)
print('Test MAE : ',model_rr_mae_without)
print('Test MAPE', mean_absolute_percentage_error(y_test_without,y_pred_rr_without))
print('Coeffficients:', model_rr_without.coef_)

# Lasso Regression
# with tfidf

from sklearn import linear_model

model_lasso = linear_model.Lasso(alpha = 1.0, max_iter = 10000, selection = 'random').fit(X_train, y_train)
model_lasso_r2 = model_lasso.score(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)
model_lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
model_lasso_mape = mean_absolute_percentage_error(y_test, y_pred_lasso)

print('R^2 : ',model_lasso_r2)
print('Test MAPE :', model_lasso_mape)
print('Test MAE : ',model_lasso_mae)
print('Coeffficients:', model_lasso.coef_)

# without tfidf

clf_without_lasso = LassoCV(alphas=alphas).fit(X_train_without, y_train_without)
clf_without_lasso.alpha_

model_lasso_without = linear_model.Lasso(alpha = clf_without_lasso.alpha_, max_iter = 10000).fit(X_train_without, y_train_without)
model_lasso_r2_without = model_lasso_without.score(X_train_without, y_train_without)
y_pred_lasso_without = model_lasso_without.predict(X_test_without)
model_lasso_mae_without = mean_absolute_error(y_test_without, y_pred_lasso_without)
model_lasso_mape_without = mean_absolute_percentage_error(y_test_without,y_pred_lasso_without)

print('R^2 : ',model_lasso_r2_without)
print('Test MAE : ',model_lasso_mae_without)
print('Test MAPE', model_lasso_mape_without)
print('Coeffficients:', model_lasso_without.coef_)

# RandomForestRegression
# GridSearch CV

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
grid_params = {'n_estimators':[x for x in range(50,500,50)],
               'max_features':[x for x in range(1400,1700,100)],
               'max_depth':[x for x in range(15,75,15)]}
grid_rf = GridSearchCV(model_rf, grid_params, n_jobs=-1)
grid_rf.fit(X_train, y_train)
end = pd.datetime.now()
grid_rf.best_params_

# with tfidf

model_rf = RandomForestRegressor(max_features = 1400, max_depth = 60, n_estimators = 350, random_state = 0)
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
model_rf_r2 = model_rf.score(X_train, y_train)
model_rf_mae = mean_absolute_error(y_test,y_pred_rf)
model_rf_mape = mean_absolute_percentage_error(y_test, y_pred_rf)
print('R^2 : ', model_rf_r2)
print('Test MAPE', model_rf_mape)
print('Test MAE : ',model_rf_mae)

# without tfidf
model_rf_without = RandomForestRegressor(max_features = 50, max_depth = 20, n_estimators = 500, random_state = 0).fit(X_train_without,y_train_without)
pred_rf_without = model_rf_without.predict(X_test_without)
model_rf_r2_without = model_rf_without.score(X_train_without, y_train_without)
model_rf_mae_without = mean_absolute_error(y_test_without,pred_rf_without)
model_rf_mape_without = mean_absolute_percentage_error(y_test_without,y_pred_lr_without)
print('R^2 : ', model_rf_r2_without)
print('Test MAPE : ', model_rf_mape_without)
print('Test MAE : ',model_rf_mae_without)


# Gradient Boosting Regression
# with tfidf

from sklearn import ensemble
parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[x for x in range(5,25,5)],
    "n_estimators":[x for x in range(500,1500,250)]
    }

clf = GridSearchCV(ensemble.GradientBoostingRegressor(), parameters, cv=4, n_jobs=-1)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.best_params_)

gb_reg = ensemble.GradientBoostingRegressor(learning_rate = 0.1, max_depth = 10, n_estimators = 1000).fit(X_train, y_train)
gb_reg_r2 = gb_reg.score(X_train, y_train)
pred_gb_reg = gb_reg.predict(X_test)
gb_reg_mae = mean_absolute_error(y_test,pred_gb_reg)
gb_reg_mape = mean_absolute_percentage_error(y_test,pred_gb_reg)
print('R^2 : ', gb_reg_r2)
print('Test MAPE : ', gb_reg_mape)
print('Test MAE : ',gb_reg_mae)

# without tfidf

parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[x for x in range(5,25,5)],
    "n_estimators":[x for x in range(500,1500,250)]
    }

clf = GridSearchCV(ensemble.GradientBoostingRegressor(), parameters, cv=4, n_jobs=-1)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.best_params_)

gb_reg_without = ensemble.GradientBoostingRegressor().fit(X_train_without, y_train_without)
gb_reg_r2_without = gb_reg_without.score(X_train_without, y_train_without)
pred_gb_reg_without = gb_reg_without.predict(X_test_without)
gb_reg_mae_without = mean_absolute_error(y_test_without,pred_gb_reg_without)
gb_reg_mape_without = mean_absolute_percentage_error(y_test_without,pred_gb_reg_without)

print('R^2 : ', gb_reg_r2_without)
print('Test MAPE :', gb_reg_mape_without)
print('Test MAE : ',gb_reg_mae_without)


# Statistical significance testing for models:
# Ridge Regression

rr_pred = model_rr.predict(X_test)
rr_diff = abs(np.array(y_test) - np.array(rr_pred))

baseline = [np.mean(y_test) for i in range(len(y_test))]
base_diff = abs(np.array(y_test) - np.array(baseline))

stats.ttest_ind(rr_diff,base_diff)
# t-statistic=-0.2635540886223598, pvalue=0.7921592161385214

# Random Forest Regressor
rf_pred = model_rf.predict(X_test)
rf_diff = abs(np.array(y_test) - np.array(rf_pred))

stats.ttest_ind(rf_diff,base_diff)
# t-statistic=-16.374360122787092, pvalue=1.2463344956887265e-55
# Gradient Boosted Regressor
gb_pred = gb_reg.predict(X_test)
gb_diff = abs(np.array(y_test) - np.array(rf_pred))

stats.ttest_ind(gb_diff,base_diff)
# t-statistic=-17.955430741957667, pvalue=1.549944221665107e-65
