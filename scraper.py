# !pip install msedge-selenium-tools selenium==3.141

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import time
import pandas as pd
import numpy as np

from msedge.selenium_tools import Edge, EdgeOptions

#launch Microsoft Edge (EdgeHTML)
driver = Edge(executable_path='msedgedriver.exe')

#list of all job titles
list_1 = ['mechanical engineer','cashier','professor','civil engineer','business analyst','chemical engineer'\
         ,'data analyst','data scientist','business intelligence engineer'\
         ,'biologist','actuarial analyst','fashion designer','nuclear engineer'\
         ,'Financial Advisor','HR manager','Financial Analyst','Marketing manager'\
        ,'Business consultant','General manager','Product manager','Project manager'\
        ,'Operations manager','Accounting manager','Public relations manager'\
        ,'Electrical design engineer','Instrumentation engineer','Product engineer'
        ,'Transmission engineer' ,'Computer engineer','Hardware design engineer','Network systems architect']

def get_jobs(position='data scientist', pages=1):
      
    search_job = driver.find_element_by_xpath('//input[@id="as_and"]')
    search_job.send_keys(position)

    display_limit = driver.find_element_by_xpath('//select[@id="limit"]//option[@value="50"]')
    display_limit.click()

    sort_option = driver.find_element_by_xpath('//select[@id="sort"]//option[@value="date"]')
    sort_option.click()
    
    loc = driver.find_element_by_xpath('//*[@id="where"]')
    loc.clear()
    
    age_option = driver.find_element_by_xpath("//select[@id='fromage']/option[1]")
    age_option.click()

    search_button = driver.find_element_by_xpath('//*[@id="fj"]')
    search_button.click()
    
#     close_popup = driver.find_element_by_xpath('//*[@id="popover-x"]/button')
#     close_popup.click()
    
    
    titles=[]
    companies=[]
    locations=[]
    links =[]
    reviews=[]
    salaries = []
    
    
    for i in range(0,pages):

        job_card = driver.find_elements_by_xpath('//div[contains(@class,"clickcard")]')

        for job in job_card:

            try:
                review = job.find_element_by_xpath('.//span[@class="ratingsContent"]').text
            except:
                review = "None"
            reviews.append(review)

            try:
                salary = job.find_element_by_xpath('.//span[@class="salaryText"]').text
            except:
                salary = "None"
        #.  tells only to look at the element       
            salaries.append(salary)

            try:
                location = job.find_element_by_xpath('.//span[contains(@class,"location")]').text
            except:
                location = "None"
        #.  tells only to look at the element       
            locations.append(location)

            titles.append(job.find_element_by_xpath('.//h2[@class="title"]//a').get_attribute(name="title"))
            links.append(job.find_element_by_xpath('.//h2[@class="title"]//a').get_attribute(name="href"))
            companies.append(job.find_element_by_xpath('.//span[@class="company"]').text)


        try:
            next_page = driver.find_element_by_xpath('//a[@aria-label={}]//span[@class="pn"]'.format(i+2))
            next_page.click()
        except:
            next_page = driver.find_element_by_xpath('//a[@aria-label="Next"]//span[@class="np"]')
            next_page.click()

        print("Page: {}".format(str(i)))
        
    df=pd.DataFrame()
    df['Title']=titles
    df['Company']=companies
    df['Location']=locations 
    df['Link']=links
    df['Review']=reviews
    df['Salary']=salaries
    
    return df

df = {}			# empty dictionary to store data for each job title

print('Scraping Data...')
for job in list_1:
    print('Scraping: ',job)
    driver = Edge(executable_path='msedgedriver.exe')    
    driver.get('https://indeed.com')

    initial_search = driver.find_element_by_xpath('//*[@id="whatWhereFormId"]/div[3]/button')
    initial_search.click()

    advanced_search = driver.find_element_by_xpath('//*[@id="jobsearch"]/table/tbody/tr/td[4]/div/a')
    advanced_search.click()    
    
    try:
        df[job] = get_jobs(position = job,pages = 10)
    
    except:
        print('Error in: ',job)
        continue

df_1 = pd.concat(df.values())

print('Scraping Job descriptions...')
descriptions = []
driver = Edge(executable_path='msedgedriver.exe')    

for i,link in enumerate(df_1['Link']):
    try:
        driver.get(link)
        jd = driver.find_element_by_xpath('//div[@id="jobDescriptionText"]').text
        descriptions.append(jd)
        print(i)
    except:
        descriptions.append('None')
        continue
df_1['job description'] = descriptions

print('Scraping Done...')