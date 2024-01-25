#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center; background-color: blue;; color: white; padding: 20px 0; font-size: 40px; font-weight: bold; border-radius: 10px ; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2);">
#   ANALYSIS OF DATA PROFESSIONS
# </div>
# 

# <div style='background-color: #fff7f7;padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ About Dataset</b></font>
# </div>

# <p style='font-size:16px; padding:10px;'>This dataset aims to shed light on the salary statistics of employees in the Data field. It will focus on various aspects of employment, including work experience, job titles, and company locations. This dataset provides valuable insights into salary distributions within the industry.

# <div style='background-color: #fff7f7;padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Objective of Analysis</b></font>
# </div>

# <p style='font-size:16px; padding:10px; border-bottom: 2px solid'>
# This notebook aims at:
#     <ul>
#       <li> Data processing</li>
#       <li> Practice using libraries to visualize data</li>
#       <li> Visualize data, provide explanations about the correlation between attributes</li>
#       <li> Draw meaningful conclusions and insights

# ## <div style="background-color:  rgb(71, 65, 65) ; color: white; padding: 15px; line-height:1;border-radius:1px; text-align: center; font-size: 25px; border-radius: 8px;  ">1. IMPORTING LIBRARIES AND DATA</div>

# In[49]:


#Importing of libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization
from scipy.stats import shapiro #Test for normality
from scipy.stats import kruskal #Hypothesis Test


# In[50]:


#Read data
data_salary = pd.read_csv('D:/Project_FoM/Analysis/Data Science Jobs Salaries.csv')


# ## <div style="background-color:  rgb(71, 65, 65) ; color: white; padding: 15px; line-height:1;border-radius:1px; text-align: center; font-size: 25px; border-radius: 8px;  ">2. EXPLORATORY DATA ANALYSIS ( EDA )</div>

# <div style='background-color: #fff7f7;padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ View Dataset</b></font>
# </div>

# In[51]:


#Viewing part of the data
data_salary.head()


# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>  Examing the Dataset</b></font>
# </div>
# 

# In[52]:


#Identifying all column headers
data_salary.columns


# In[53]:


#Identifying the various job titles
jobs_available=data_salary['job_title'].unique()
sum_unique=data_salary['job_title'].value_counts().count()
print(sum_unique, ' jobs can be found in the dataset')


# In[54]:


#Identifying experience levels
data_salary['experience_level'].unique()


# `Experience_level`: 
#    - EN: Entry-level / Junior
#    - MI: Mid-level / Intermediate
#    - SE: Senior-level / Expert
#    - EX: Executive-level / Director

# In[55]:


data_salary['employment_type'].unique()


# `Employment_type`: 
#    - FT: Full-Time
#    - PT: Part-Time
#    - CT: Contractor
#    - FL: Freelancer

# In[56]:


#Identifying the company size
data_salary['company_size'].unique()


# `Company_Size`: 
#    - L: Large
#    - M: Medium
#    - S: Small

# In[57]:


#Identifying the remote ratio
data_salary['remote_ratio'].unique()


# `Remote_ratio`: 
#    - 0: None remote
#    - 50: Hybrid
#    - 100: Fully remote

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Cleaning the Dataset</b></font>
# </div>
# 

# <p style='font-size:16px; padding:10px; border: 2px solid'>Combining the 3 jobs we are working with.ie. Data Science, data analysis and data engineer.</p>

# In[58]:


data_scientist=data_salary[data_salary['job_title']=='Data Scientist']
data_analyst= data_salary[data_salary['job_title']=='Data Analyst']
data_engineer=data_salary[data_salary['job_title']=='Data Engineer']
work_data=pd.concat([data_scientist,data_analyst,data_engineer],axis=0)


# In[59]:


#  Replace values in work_year column and change data type
work_data['work_year'] = work_data['work_year'].replace('2021e', '2021')
work_data['work_year'] = work_data['work_year'].astype('int64')

#  Replace values in experience-level column
work_data['experience_level'] = work_data['experience_level'].replace('EN', 'Entry-Level')
work_data['experience_level'] = work_data['experience_level'].replace('EX', 'Experienced')
work_data['experience_level'] = work_data['experience_level'].replace('MI', 'Mid-Level')
work_data['experience_level'] = work_data['experience_level'].replace('SE', 'Senior-Level')
# Replace values in employment_type column
work_data['employment_type'] = work_data['employment_type'].replace('FT', 'Full-Time')
work_data['employment_type'] = work_data['employment_type'].replace('CT', 'Contractor')
work_data['employment_type'] = work_data['employment_type'].replace('FL', 'Freelancer')
work_data['employment_type'] = work_data['employment_type'].replace('PT', 'Part-Time')
# Replace values in Company size column
work_data['company_size'] = work_data['company_size'].replace('L', "Large")
work_data['company_size'] = work_data['company_size'].replace('M', "Medium")
work_data['company_size'] = work_data['company_size'].replace('S', "Small")
# Replace values in remote ratio column and change data type
work_data['remote_ratio'] = work_data['remote_ratio'].replace(0, "None remote")
work_data['remote_ratio'] = work_data['remote_ratio'].replace(50, "Hybrid")
work_data['remote_ratio'] = work_data['remote_ratio'].replace(100, "Fully remote")
work_data['remote_ratio'] = work_data['remote_ratio'].astype(object)


# New data
work_data=work_data.reset_index(drop = True)
work_data.head()


# <div style='background-color: #fff7f7;padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Checking for null values</b></font>
# </div>
# 
# 

# In[60]:


work_data.info()
data_salary[data_salary.isnull()].count()


# <div style='background-color: #fff7f7;  padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Trend of the Data</b></font>
# </div>
# 

# In[61]:


#Year employees joined the domain
work_data.groupby('work_year')['work_year'].count().plot.pie(autopct='%1.1f%%')
plt.title('Comparison of Years')
plt.style.use('default')


# In[63]:


#plottings work year by salary
sns.lineplot(data =work_data ,x = 'work_year', y = 'salary_in_usd')
plt.title('Salary Trend ', fontweight='bold')
plt.legend(['Salary'])
plt.show()
plt.style.use('ggplot')


# <div style='border: 2px solid; padding:20px; font-size: 16px;'>
#     <ul>
#         <li>As the year increase from <b>2020</b> to <b>2021</b>, the job demand increases by almost 50% to that of the previous year.
#         </li>
#         <li>This leads to a decrease in the salary relatively to the increase in year.</li>
#         <li><b>Work_year</b> and <b>salary_in_usd</b> thus have a negative correlation</li>
#     </ul>
# </div>

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦Summary Statistics</b></font>
# </div>
# 

# In[64]:


#Finding pearson correlation
work_data[['work_year','salary','salary_in_usd']].corr(method='pearson')


# * `Salary_in_usd` has a negative correlation of approximately -0.11 with `salary.` This suggests that as `Salary_in_usd` increases, `salary` tends to decrease slightly. This is accurate since the conversion rate of various currencies to USD are not equal.
# * `Salary_in_usd` and `work_year` have a negative correlation of around -0.046. This indicates that as `Salary_in_usd` goes up, `work_year` also tends to decrease. This was proven in our figure when we were analyzing  the trend.
# 

# In[65]:


print('--Summary statistics of the 3 job types--')
work_data.groupby('job_title')['salary_in_usd'].describe(include='all').T


# <p style='font-size:16px; padding:10px; border: 2px solid'><b>Data Engineer</b> records the highest average salary with a count of 38 people.</p>

# In[66]:


print('--Further information after grouping by experience level and work year--')

work_data.groupby(['job_title', 'experience_level','work_year'])['salary_in_usd'].describe()


# <div style='background-color: #fff7f7;  padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Comparing the 3 job types</b></font>
# </div>
# 

# In[67]:


plotdata = work_data['job_title'].value_counts()
plotdata.plot.pie(autopct='%1.1f%%')
plt.title('Total number of Employees')
print(plotdata)


# <p style='font-size:16px; padding:10px; border: 2px solid'>In our dataset, we notice that data scientist have half the count of our cleaned dataset. This implies that, data science occupies a huge proportion  than the other 2 data field. </p>

# In[68]:


work_data.boxplot(column='salary_in_usd',by='job_title')
plt.ylabel('Salary_in_usd')
plt.title('')


# <p style='font-size:16px; padding:10px; border: 2px solid'>Overall, the salaries for data analysts, data engineers, and data scientists are all relatively high. Data Engineers have the highest median salary, followed by data analyst and then data scientist. There are a few possible explanations for these salary differences. One of these is the fact that data scientists are in higher demand than data analysts or data engineers. There are a few outliners in the data analyst and data scientist field indicating earnings beyond the maximum and the minimum salary.<br>
# This could have been caused by various factors such as individuals being extremely good at their work or on the other hand being bad at it. It could have also been as a result of improper record taking. A further analysis of the data will explain it further.</p>
# 

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Comparison by Experience Levels</b></font>
# </div>

# In[69]:


exp=work_data.groupby([ 'experience_level','job_title'])['salary_in_usd'].mean().unstack().plot.bar()
plt.ylabel('Average_Salary')
plt.xlabel('Experience_Level')
plt.style.use('default')
plt.title('Average Salary by Experience Level')
plt.xticks(rotation=0)
plt.style.use('ggplot')

#Place values above chart
for container in exp.containers:
    exp.bar_label(container, label_type="edge", color="black",
                 padding=6, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': 'black'})


# <div style='border: 2px solid; padding:20px ; font-size: 16px;'>
#     <ul>
#         <li>In the Entry - Level, Data Analysis receive the highest average income
#         </li>
#         <li>In the Mid - Level, Data Engineer receive the highest average income.</li>
#         <li>In the Senior - Level, Data Scientist receive the highest average income.</li>
#     </ul>
# </div>

# In[70]:


work_data[work_data['company_location']=='SG']


# In[71]:


work_data['experience_level'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Comparison of Experience Levels')


# <p style='font-size:16px; padding:10px; border: 2px solid'> The skill set of most workers lie in the Mid - Level Tier

# In[72]:


exp_level = 'Mid-Level' #Check for this experience level
salary_range = (60000, 100000)  # check this range
range_level = work_data[(work_data['experience_level'] == exp_level) &
                 (work_data['salary_in_usd'] >= salary_range[0]) &
                 (work_data['salary_in_usd'] <= salary_range[1])]
available = range_level['job_title'].value_counts().reset_index() # Count for each job title
available.columns = ['Job Title', 'Count'] # Change headers

p=sns.barplot(y='Count', x='Job Title', data=available, palette = 'rainbow')
plt.ylabel('Vacancy')
plt.xlabel('Job Titles')
plt.title(f'Available Vacancies for {exp_level} Candidates \n Salary Range {salary_range[0]} - {salary_range[1]}', fontsize=14, fontweight='bold')
plt.style.use('default')

for container in p.containers:
    p.bar_label(container, padding=-10,fontsize=22,
                 bbox={'boxstyle': 'circle,pad=0.3', 'facecolor': 'white', 'edgecolor': 'white'})


# <div style='border: 2px solid; padding:20px; font-size: 16px; text-align: center'><i>Mid-Level vacancy available</i>
#     <p></p>
# <ul style="list-style-type: disc; padding-left: 20px; text-align: left">
#         <li>Data Analyst - 5 job openings</li>
#         <li>Data Scientist - 9 job openings</li>
#         <li>Data Engineer -7 job openings</li>
#    </ul>

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Further Statistics of location</b></font>
# </div>

# In[73]:


#Distribution of employees based on employees residence
in_loc = work_data[work_data['employee_residence']==work_data['company_location']]
out_loc = work_data[work_data['employee_residence']!= work_data['company_location']]
loc = in_loc.count()['work_year'], out_loc.count()['work_year']
plt.pie(loc,labels=['Employees Live in Company Location', 'Employees Live elsewhere'], autopct='%1.1f%%',explode=[0,0.2])
plt.style.use('default')
plt.title('Comparison of Location')


# In[74]:


loc


# In[76]:


remote=work_data.groupby('remote_ratio')['remote_ratio'].count()
ratio=sns.barplot(y=remote.index,x=remote.values)
plt.xlabel('Number of people')
plt.title('Comparison of Remote Location')
for container in ratio.containers:
    ratio.bar_label(container, label_type="edge", color="black",
                 padding=-13,fontsize=12,bbox={'boxstyle': 'rarrow', 'facecolor': 'white', 'edgecolor': 'black'})


# <p style='font-size:16px; padding:10px; border: 2px solid'>Despite a huge population living in the country of their company's location, most prefer to work remotely</p>

# In[78]:


# Group the data by company_location and calculate the mean salary for each location
salary_location = work_data.groupby('company_location')['salary_in_usd'].mean().reset_index()

# Sort the locations by average salary in descending order
salary_location = salary_location.sort_values(by='salary_in_usd', ascending=False)
# Create a bar chart to visualize average salaries by country
location = sns.barplot(x='salary_in_usd', y='company_location', data=salary_location.head(5))
plt.title('Top 5 Average Salaries by Location', fontweight='bold' )
plt.xlabel('Average Salary (USD)')
plt.ylabel('Location')
plt.style.use('ggplot')
for container in location.containers:
    location.bar_label(container, bbox = {'boxstyle': 'circle', 'edgecolor': 'red', 'facecolor': 'white'},
                label_type="center",
               )


# <div style='border: 2px solid; padding:20px; font-size: 16px; text-align: center'><i>Top 5 Countries</i>
#     <br>
#     <ul style="list-style-type: disc; padding-left: 20px; text-align: left">  <li><b>Illinois (IL)</b> records the highest average data salary at approximately <i>119353 USD</i>.</li>
#         <li><b>United States (US)</b> and <b>Canada(CA)</b> also offers a competitive average salaries, with approximately <i>111994 USD and 84962.2 USD</i>, respectively.</li>
#         <li><b>Great Britain (GB)</b> and <b>Austria (AT)</b> round up the top 5 locations with varying average salaries of <i>81016 USD and 75784 USD</i> .</li>

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Further Statistics</b></font>
# </div>

# In[79]:


#Group data by 'employment_type' and calculate the average salary for each type
emp_salary = work_data.groupby(['job_title','remote_ratio','employment_type'])['salary_in_usd'].mean()

emp = emp_salary.plot(kind='bar',color='gray')
plt.title('Average Salary by Employment Type', fontsize=12, fontweight='bold')
plt.xlabel('Employment Type')
plt.ylabel('Average Salary (USD)')
plt.style.use('ggplot')
emp_salary = work_data.groupby(['job_title','remote_ratio','employment_type'])['salary_in_usd'].mean()

emp = emp_salary.plot(kind='bar',color='gray')
plt.title('Average Salary by Employment Type', fontsize=12, fontweight='bold')
plt.xlabel('Employment Type')
plt.ylabel('Average Salary (USD)')
plt.style.use('ggplot')

# labels on chart
for container in emp.containers:
    emp.bar_label(container, label_type="edge", color="black",padding=-13,
                 bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'gray'})


# <p style='font-size:16px; padding:10px; border: 2px solid'>Grouping the <b>remote_ratio</b> by <b>employment_type</b>, those working <b>Fully remote and Full-Time</b> obtain the highest average salary with <b>91298.2 USD</b>. This is earned by <b>Data Engineers</b>.

# In[80]:


company_size_salary = work_data.groupby('company_size')['salary_in_usd'].mean()
p = sns.barplot(y=company_size_salary.index, x=company_size_salary.values)
plt.title('Average Salary by Company Size',  fontweight='bold')
plt.ylabel('Company Size')
plt.xlabel('Average_salary')

# labels on chart
for container in p.containers:
    p.bar_label(container, label_type="edge", color="black",
                 padding=-45,bbox={'boxstyle': 'larrow', 'facecolor': 'white', 'edgecolor': 'white'})


# <p style='font-size:16px; padding:10px; '> Large companies tend to pay a higher average of about 87500 to their employees than that of Medium and Small companies.</p>

# ## <div style="background-color:  rgb(71, 65, 65) ; color: white; padding: 15px; line-height:1;border-radius:1px; text-align: center; font-size: 25px; border-radius: 8px;  ">3. HYPOTHESIS TEST </div>

# <p style='font-size:16px; padding:10px; text-align: center'><i>TESTING OF THE HYPOTHESIS</i>
#     <br><br>
#     <b>Null hypothesis: </b>Data scientist earn a high amount of money in USD compared to the other parallel professions.
#     <br>
#     <b>Alternate hypothesis: </b>Data scientist do not earn a high amount of money in USD compared to the other parallel professions.

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Checking for normality whith Histogram</b></font>
# </div>

# In[81]:


sns.histplot(work_data['salary_in_usd'], kde=True,bins=20)
plt.title("Salary Distribution of Data Professionals")
plt.style.use('ggplot')
plt.ylabel('Frequency')


# <p style='font-size:16px; padding:10px; border: 2px solid'>The histogram is skewed to the right and thus, the data is not symmetric. This implies that, it is not normally distributed</p>

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Checking for normality with Shapiro Test </b></font>
# </div>

# In[82]:


# Check for normality
x=data_scientist['salary_in_usd']
y=data_analyst['salary_in_usd']
z=data_engineer['salary_in_usd']
stat, p=shapiro(x)
print(' Stat = ',round(stat,2) ,'\n P-value = ', round(p,8))
if p > 0.05:
    print(' The data is normally distributed')
else:
    print(' The data is not normally distributed')


# <p style='font-size:16px; padding:10px; border: 2px solid'>Since the data is not not normally distributed, we therefore cannot use parametric test for the hypothesis test. We will have to use non-parametric test to check the hypothesis.</p>

# <div style='background-color: #fff7f7; padding :5px; border-radius: 8px 8px 0 0;'>
#     <font size="+2" color="salmon "><b>♦ Using non-parametric test (Kruskal Wallis)</b></font>
# </div>

# In[83]:


statistics , p_value=kruskal(x,y,z)
print(' Stat = ',round(statistics,2) ,'\n P-value = ', round(p_value,2))
if p_value < 0.05:
    print(' We reject the null hypothesis. \n There is significant evidence to suggest difference between the datasets.')
else:
    print(' We fail to reject the null hypothesis. \n There is no significant evidence to suggest differences between the salaries.')


# <p style='font-size:16px; padding:10px; border: 2px solid'>This means that our null hypothesis holds. That is:
#     <b>data scientist earn a high amount of money in USD compared to the other parallel professions.</b><p> 

# <div style="text-align: center; background-color: gray;; color: white; padding: 20px ; font-size: 30px; border-radius: 10px ; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2); text-align:right"><i>The end</i><p style="text-align: right; font-size: 15px">By: <i>Musah Faridu Oubda</i></p>
# 
# </div>
# 
