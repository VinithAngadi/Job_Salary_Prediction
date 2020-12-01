# Job Salary Prediction
Our project aims to assist job seekers negotiate salaries by predicting what salary to expect for a certain job description at a given location. There are tens of thousands of jobs on online job boards that invite job seekers to apply for jobs without providing an estimate of the compensation they provide – this was further validated when we were scraping for salaries online. This poses a lot of problems for job seekers, especially new graduates like us, applying for a job role or when trying to negotiate the compensation. These problems are what we seek to solve in our project, where we train machine learning models to predict salaries for a job posting at a certain location in the United States. We, being job seekers, are facing this problem every day in our job hunt which gives us a strong motivation to build an efficient and reliable model to help us in finding the right compensation for our jobs.

Data Collection: 
In order to predict the salary from the set of predictors, we needed to collect a lot of data to train the machine learning models. We wanted to work with raw data from the real-world, hence we chose to scrape data off of Indeed.com, a popular job board with variety of jobs. We used Selenium web scraper with search strings ranging from ‘Software Engineer’, ‘Data Scientist’, ‘Project Manager’, ‘Mechanical Engineer’, ‘Accountant’ to ‘Cashier’ to scrape 30,000 job postings. The final scraped data was a CSV file containing the job listings with job descriptions and location. 

Data Cleaning & Preprocessing:
The meat of information about any job listing is present in its job description which prompted us to thoroughly clean, process and extract features from the job description column. The salary column too was highly inconsistent with different formats of salary like hourly, monthly and yearly which had to be standardized. The cleaning was performed using regular expressions and stopwords from NLTK library. 
•	Job Description column cleaned by removing punctuation, new line characters, extra spaces/tabs and stopwords. 
•	Salary column standardized by converting all formats to salary (in $) per year using RegEx and conversion rules. 
•	States extracted from job location column and standardized to remove spelling errors. 
 
•	With a pre-defined set of vocabulary, converted the job description text to a Term Frequency – Inverse Document Frequency (TF-IDF) vector.
•	Added cost index and purchasing power parity of each state for the job listings.
The final data after scraping, cleaning and preprocessing before vectorization looks like below:
 

Topic Modeling:
The domain of the job is a very important factor in predicting the salary, since the domain tells us the required skills and these skills usually directly influence the resultant salary. This posed an unsupervised learning problem to us since the job domain was not labeled and we needed to group and identify the job field from the job description. We implemented topic modeling using non-negative matrix factorization where each job listing was grouped into one of the 10 topics. We manually annotated and provided labels to these 10 topics by looking at the top words in each of the topic. These topics were added to the data table as one of the predictors for every job listing. Two topics with their top 30 words and annotation are shown below:
 
The Model:
After the cleaning, preprocessing and topic modeling steps, our data had job location, cost index, purchasing power parity, TF-IDF vector of job description and the domain obtained from topic modeling. With one-hot encoding of the categorical features and TF-IDF vector, the final ‘X’ vector of all the predictors consisted of 7241 columns. The ‘y’ or target of our prediction was the salary for a job posting – making it a regression problem. We used StandardScaler to normalize cost index, purchasing power parity and the TF-IDF vector before training our models. 
We initially tried a Linear Regression model, but the model was too ‘simple’, and a straight line couldn’t capture the relationship between X and y. Also, due to multicollinearity induced from the TF-IDF vector, the Linear Regression model had large errors. In order to take care of multicollinearity and regularize the model, we implemented L2 regularization with Ridge Regression. This gave us a much-improved linear model with closer predictions of the target salary.
However, since linear models assume the presence of a linear relationships between the predictors and target, they weren’t able to provide the best of results. We needed a more ‘complex’ model capable of capturing such a relation. Random Forests and Gradient Boosted Trees are great for these kinds of problems since they learn freely based off of the data without any assumed relationship between the predictor and target variables. Also, these tree-based models avoid overfitting by aggregating results (in case of Random Forests) and incremental learning (in case of Gradient Boosted Trees), which make them good models for our regression problem.

Flask API:
	To enable ease of access and use of our project, we hosted it on a server with Flask API. The user is prompted to enter the job description and location and the predicted salary is displayed on the webpage. The model communicates with the user via the Flask API.
