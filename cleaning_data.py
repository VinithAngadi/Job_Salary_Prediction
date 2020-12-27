import pandas as pd
import re

df = pd.read_csv('Cleaned_data_1.csv')
df_1 = pd.read_csv('df_scrapped_JD_3.csv')
df_2 = pd.read_csv('df_scrapped_JD_4.csv')
df_3 = pd.read_csv('df_3_200.csv')
df_4 = pd.read_csv('df_1_200.csv')
df_5 = pd.read_csv('df_2_200.csv')

df = pd.concat([df, df_1, df_2, df_3, df_4, df_5], ignore_index = False)

def get_states(location_list):

    states = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado",
               "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii",
               "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
               "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana",
               "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico",
               "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island",
               "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands",
               "Vermont", "Washington State", "Wisconsin", "West Virginia", "Wyoming", "AL", "AK", "AZ", "AR", "CA",
               "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
               "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
               "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SC",
               "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "Remote", "United States"]


    states_list = []
    for loc in location_list:
        address_split = loc.split(' ')
        flag = 0
        i = 0
        for word in address_split:
            if word in states:
                if len(word) == 2:
                    states_list.append(word)
                    flag = 1
                    i += 1

        if flag == 0:
            if loc in states:
                s = ''
                if loc == 'United States':
                    states_list.append('R')
                else:
                    for w in loc.split(' '):
                        s += (w[0])
                    flag = 1
                    states_list.append(s)
    return states_list


def get_salary(salaries_list):
    salary_list = []
    period_list = []
    for str_sal in salaries_list:
        str_sal_split = str_sal.split(' ')
        salaries = re.findall('(?:[\£\$\€]{1}[,\d]+.?\d*)', str_sal)
        salary_num = 0
        for sal in salaries:
            sal = sal.strip()
            sal = sal.replace(',','')
            sal = sal.replace('$','')
            salary_num += int(float(sal))
        salary = salary_num/len(salaries)
        if str_sal_split[-1] == '++':
            pay_period = str_sal_split[-2]
        else:
            pay_period = str_sal_split[-1]
        period_list.append(str_sal_split[-1])
        if pay_period == 'hour':
            salary_list.append(salary * 40 * 52)
        elif pay_period == 'month':
            salary_list.append(salary * 12)
        elif pay_period == 'day':
            salary_list.append(salary * 365)
        elif pay_period == 'week':
            salary_list.append(salary * 52)
        elif pay_period == 'year':
            salary_list.append(salary)
        else:
            print(str_sal)
    return salaries_list



def clean_job_desc(df_job_desc):

    # Convert to lower case
    df_job_desc['job_description_lower'] = [x.lower() for x in df['job_description']]

    # Remove punctuations
    df_job_desc['job_description_lower'] = [re.sub(r"[^a-zA-Z0-9]", " ",x) for x in df['job_description_lower']]

    return df_job_desc





df.drop(['Unnamed: 0', 'Company','Link','Review'], axis = 1, inplace = True)
df = df[df['Location'] != 'None']
df = df[df['Salary'] != 'None']
df.dropna(axis = 0, inplace = True)
df['States'] = get_states(list(df['Location']))
df['Salary'] = get_salary(list(df['Salary']))
df = clean_job_desc(df)

df.to_csv('Final_cleaned_data.csv')
