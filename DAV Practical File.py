
#Five Girls respectively as values associated with these keys
#Original dictionary of lists:
#{'Boys': [72, 68, 70, 69, 74], 'Girls': [63, 65, 69, 62, 61]}
#From the given dictionary of lists create the following list of dictionaries:

#[{'Boys': 72, 'Girls': 63}, {'Boys': 68, 'Girls': 65}, {'Boys': 70, 'Girls': 69}, {'Boys': 69, 'Girls': 62}, {‘Boys’:74,
#‘Girls’:61]


#Q1:
mydict = {'Boys': [72, 68, 70, 69, 74], 'Girls': [63, 65, 69, 62, 61]}

def list_of_dicts(values):
    keys = values.keys()
    vals = zip(*[values[k] for k in keys])
    result = [dict(zip(keys, v)) for v in vals]
    return result

print(f"The original dictionary: {mydict}")
mydictfinal = list_of_dicts(mydict)
print(f"The reultant dictionary: {mydictfinal}")

"""Write programs in Python using NumPy library to do the following:
a. Compute the mean, standard deviation, and variance of a two dimensional random integer array
along the second axis.

b. Get the indices of the sorted elements of a given array.

  a. B = [56, 48, 22, 41, 78, 91, 24, 46, 8, 33]

c. Create a 2-dimensional array of size m x n integer elements, also print the shape, type and data
type of the array and then reshape it into nx m array, n and m are user inputs given at the run time.

d. Test whether the elements of a given array are zero, non-zero and NaN. Record the indices of
these elements in three separate arrays.
"""

#Q2 a
import numpy as np
#random integer array
a = np.random.randint(10, size = (3,3))

def calculate(a):
  mean = np.mean(a)
  standard_deviation = np.std(a)
  variance = np.var(a)

  return mean, standard_deviation, variance

amean, astd, avar = calculate(a)
print(f"Our 2 Dimensional Array: {a}")
print(f"Mean = {amean}")
print(f"Standard Deviation = {astd}")
print(f"Variance = {avar}")

#Q2 b
B = [56, 48, 22, 41, 78, 91, 24, 46, 8, 33]

def indices(x):
  arr =np.argsort(x, axis=-1, kind='quicksort', order=None)
  return arr

sorted_arr = indices(B)
print(f"Original array : {B}")
print(f"Indices of the sorted elements of B: {sorted_arr}")

#Q2 c

def create_arr(r,c):
  arr = np.random.randint(10, size=(r,c))
  return arr

m = int(input("Enter number of rows"))
n = int(input("Enter number of columns"))
myarr = create_arr(m,n)

4print(f"Our array(m,n): {myarr}")
#Shape
print(f'Shape of array: {myarr.shape}')
#Type
print(f"Data Type of array: {myarr.dtype}")
#Reshape
myarr1 = myarr.reshape(n,m)
print(f"Reshape array (n*m) : {myarr1}")

#Q2 d
#Our given array
arr = np.array([1, 2, 10, 50, -np.nan, 0., np.nan, 0.0, np.nan, 2,3,5, 0])

def check(arr):
  arrnan = np.isnan(arr)
  arrnon_zero = np.any(arr)
  arr_zero = bool(np.where(arr == 0))
  return arrnan, arrnon_zero, arr_zero

def create_separate_arrays(arr):
  indexnans = np.argwhere(np.isnan(arr))
  indexnon_zero = (np.nonzero(arr))
  index_zero = np.where(arr == 0)
  return indexnans, indexnon_zero, index_zero

nan_final, nonzerofinal, zerofinal = create_separate_arrays(arr)
boolnan, boolnon_zero, bool_zero = check(arr)
print(f"Our given array = {arr}")
print(f'Checking wether nan, null or non zeros exits, (boolean results)')
print(f"Nan Values: {boolnan}")
print(f"Non Zero values: {boolnon_zero}")
print(f"Zero values: {bool_zero}")
print(f"Creating separate arrays containing indices of values")
print(f"Nan Values: {nan_final}")
print(f"Non Zero values: {nonzerofinal}")
print(f"Zero values: {zerofinal}")

"""Q3.
Create a dataframe having at least 3 columns and 50 rows to store numeric data generated using a random
function. Replace 10% of the values by null values whose index positions are generated using random function.
Do the following:
a. Identify and count missing values in a dataframe.

b. Drop the column having more than 5 null values.

c. Identify the row label having maximum of the sum of all values in a row and drop that row.

d. Sort the dataframe on the basis of the first column.

e. Remove all duplicates from the first column.

f. Find the correlation between first and second column and covariance between second and third
column.

g. Detect the outliers and remove the rows having outliers.

h. Discretize second column and create 5 bins
"""

import pandas as pd
#Creating Dataframe
column1 = np.random.rand(50,1)
column2 = np.random.rand(50,1)
column3 = np.random.rand(50,1)
column1 = pd.DataFrame(column1, columns = ['A'])
column2 = pd.DataFrame(column2, columns = ['B'])
column3 = pd.DataFrame(column3, columns = ['C'])

df = pd.concat([column1, column2, column3], axis = 1)
#Replaccing 10% values with nan
for col in df.columns:
    df.loc[df.sample(frac=0.1).index, col] = pd.np.nan

#a.Identify and count missing values
boolean_isnan = df.isna()
count_na = df.isna().sum()
print(f"Identifying nan values : \n{boolean_isnan}")
print(f"Count of null values in each column: {count_na}")

#b)Drop column with more than 5 nan values
def drop_column(df):
  if df['A'].isna().sum() > 5:
    df1 = df.drop(['A'], axis = 1)
  elif df['B'].isna().sum() > 5:
    df1 = df.drop(['B'], axis = 1)
  elif df['C'].isna().sum() > 5:
    df1 = df.drop(['C'], axis = 1)

  return df

#c) Identify row with max value
n = df.shape[0]
m = df.shape[1]
max_sum = 0
for i in range(n):
    sum = df['A'][i] + df['B'][i] + df['C'][i]
    if sum > max_sum:
      max_sum = sum
index_row_max = df[(df['A'] + df['B'] + df['C']) == max_sum].index

#Q3 d
df.sort_values('A')
df

#Q3 d
df.sort_values('A')
df

#Q3 e
df['A'].drop_duplicates()

#Q3 f
#Correlation between A and B
firsttwocolumns = df.drop(['C'], axis = 1)
print(f"Correlation between A and B \n {firsttwocolumns.corr()}")
#Covariance between B and C
lasttwocolumns = df.drop(['A'], axis = 1)
print(f"\nCovariance between B and C {lasttwocolumns.cov()}")

#Q3 g
import seaborn as sns
sns.boxplot(df)

#dropping rows where outliers exist
arr_outliers = np.where(df['A'] < 0.2)[0]
df.drop(index = arr_outliers, inplace=True)

#Q3h: Discretize and create 5 bins
pd.cut(df["B"],bins=5, labels=["Very bad", "Bad", "Average", "good", "Very good"])

#Q4
import pandas as pd


# Load the Excel files into dataframes
file1_path = 'Q4excel1.xlsx'
file2_path = 'Q4excel2a.xlsx'

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# Merge dataframes and use 'Name' and 'duration' as multi-row indexes
merged_multiindex_df = pd.merge(df1, df2)

# Generate descriptive statistics for the multi-index dataframe
descriptive_stats = merged_multiindex_df.describe()

# Print the descriptive statistics
print("Descriptive statistics for multi-index dataframe:")
print(descriptive_stats)

"""
5. Taking Iris data, plot the following with proper legend and axis labels: (Download IRIS data from:
https://archive.ics.uci.edu/ml/datasets/iris or import it from sklearn.datasets)
a. Plot bar chart to show the frequency of each class label in the data.
b. Draw a scatter plot for Petal width vs sepal width.
c. Plot density distribution for feature petal length.
d. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset.


"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

# Set the style for the plots
plt.style.use('seaborn')

# a. Plot bar chart to show the frequency of each class label in the data.
class_counts = iris_data['target'].value_counts()
class_labels = iris.target_names

plt.figure(figsize=(8, 5))
plt.bar(class_labels, class_counts)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class Label')
plt.show()

# b. Draw a scatter plot for Petal width vs Sepal width.
plt.figure(figsize=(8, 5))
for class_label in iris.target_names:
    plt.scatter(iris_data[iris_data['target'] == class_label]['sepal width (cm)'],
                iris_data[iris_data['target'] == class_label]['petal width (cm)'],
                label=class_label)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Width vs Sepal Width')
plt.legend(title='Class')
plt.show()

# c. Plot density distribution for feature petal length.
plt.figure(figsize=(8, 5))
for class_label in iris.target_names:
    sns.histplot(data=iris_data[iris_data['target'] == class_label],
                 x='petal length (cm)', kde=True, label=class_label)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.title('Density Distribution of Petal Length')
plt.legend(title='Class')
plt.show()

# d. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset.
import seaborn as sns
sns.pairplot(iris_data, hue='target', diag_kind='kde', markers=["o", "s", "D"])
plt.suptitle('Pairwise Bivariate Distribution in Iris Dataset', y=1.02)
plt.show()

"""Taking Iris data, plot the following with proper legend and axis labels: (Download IRIS data from:
https://archive.ics.uci.edu/ml/datasets/iris or import it from sklearn.datasets)
a. Plot bar chart to show the frequency of each class label in the data.
b. Draw a scatter plot for Petal width vs sepal width.
c. Plot density distribution for feature petal length.
d. Use a pair plot to show pairwise bivariate distribution in the Iris Datase

"""

import pandas as pd

# Load the Excel files into dataframes

df1 = pd.read_excel('Q4excel1.xlsx')
df2 = pd.read_excel('Q4excel2a.xlsx')

# Merge dataframes and use 'Name' and 'duration' as multi-row indexes
merged_multiindex_df = pd.merge(df1, df2, on=['Name'], how='outer').set_index(['Name'])

# Generate descriptive statistics for the multi-index dataframe
descriptive_stats = merged_multiindex_df.describe()

# Print the descriptive statistics
print("Descriptive statistics for multi-index dataframe:")
print(descriptive_stats)

#Q7 Consider a data frame containing data about students i.e. name, gender and passing division:

import pandas as pd

# Create the DataFrame
data = {'Name': ['Mudit Chauhan', 'Seema Chopra', 'Rani Gupta', 'Aditya Narayan', 'Sanjeev Sahni',
                 'Prakash Kumar', 'Ritu Agarwal', 'Akshay Goel', 'Meeta Kulkarni', 'Preeti Ahuja',
                 'Sunil Das Gupta', 'Sonali Sapre', 'Rashmi Talwar', 'Ashish Dubey',
                 'Kiran Sharma', 'Sameer Bansal'],
        'Birth_Month': ['December', 'January', 'March', 'October', 'February', 'December',
                        'September', 'August', 'July', 'November', 'April', 'January',
                        'June', 'May', 'February', 'October'],
        'Gender': ['M', 'F', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M'],
        'Pass_Division': ['III', 'II', 'I', 'I', 'II', 'III', 'I', 'I', 'II', 'II', 'III', 'I', 'III', 'II', 'II', 'I']}

df = pd.DataFrame(data)

# a. Perform one hot encoding of the last two columns using get_dummies()
encoded_df = pd.get_dummies(df, columns=['Gender', 'Pass_Division'], prefix=['Gender', 'Division'])

# b. Sort the DataFrame based on "Birth Month" column
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']

encoded_df['Birth_Month'] = pd.Categorical(encoded_df['Birth_Month'], categories=month_order, ordered=True)
sorted_df = encoded_df.sort_values('Birth_Month')

print(sorted_df)

import pandas as pd

# Sample family data
data = {'Name': ['Shah', 'Vats', 'Vats', 'Kumar', 'Vats', 'Kumar', 'Shah', 'Shah', 'Kumar', 'Vats'],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
        'MonthlyIncome': [114000.00, 65000.00, 43150.00, 69500.00, 155000.00, 103000.00,
                          55000.00, 112400.00, 81030.00, 71900.00]}

family_data = pd.DataFrame(data)

# a. Calculate and display familywise gross monthly income
familywise_gross_income = family_data.groupby('Name')['MonthlyIncome'].sum()
print("Familywise Gross Monthly Income:")
print(familywise_gross_income)

# b. Calculate and display the member with the highest monthly income in a family
highest_income_member = family_data.loc[family_data.groupby('Name')['MonthlyIncome'].idxmax()]
print("\nMember with Highest Monthly Income in Each Family:")
print(highest_income_member)

# c. Calculate and display monthly income of members with income greater than Rs. 60000.00
high_income_members = family_data[family_data['MonthlyIncome'] > 60000.00]
print("\nMonthly Income of Members with Income > Rs. 60000.00:")
print(high_income_members)

# d. Calculate and display the average monthly income of female members in the Shah family
average_female_income_shah = family_data[(family_data['Name'] == 'Shah') & (family_data['Gender'] == 'Female')]['MonthlyIncome'].mean()
print("\nAverage Monthly Income of Female Members in Shah Family:")
print(average_female_income_shah)

#Q9
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('covid_19_india.csv')

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the year 2020
data_2020 = data[data['Date'].dt.year == 2020]

# Filter data for the required states
states = ['Karnataka', 'Gujarat', 'Haryana', 'Uttar Pradesh']

# Subplot for total cured cases month-wise from April 2020 to March 2021
plt.figure(figsize=(12, 6))
for state in states:
    state_data = data_2020[data_2020['State/UnionTerritory'] == state]
    state_data = state_data[state_data['Date'].dt.month >= 4]
    state_data = state_data[state_data['Date'].dt.month <= 12]
    state_data = state_data.groupby(state_data['Date'].dt.month)['Cured'].sum()
    state_data.plot(kind='line', label=state)
plt.xlabel('Month')
plt.ylabel('Total Cured Cases')
plt.title('Total Cured Cases Month-wise (April 2020 - December 2020)')
plt.legend()
plt.show()

# Filter data for the months of May 2020 and May 2021
may_2020 = data[(data['Date'].dt.year == 2020) & (data['Date'].dt.month == 5)]
may_2021 = data[(data['Date'].dt.year == 2021) & (data['Date'].dt.month == 5)]

# Stacked bar plot for deaths comparison in May 2020 and May 2021
deaths_comparison = pd.concat([may_2020, may_2021])
deaths_comparison = deaths_comparison.groupby(['State/UnionTerritory', 'Date'])['Deaths'].sum().unstack()
deaths_comparison.plot(kind='bar', stacked=True)
plt.xlabel('States')
plt.ylabel('Total Deaths')
plt.title('Deaths Comparison in May 2020 and May 2021')
plt.legend(['May 2020', 'May 2021'])
plt.show()

# Filter data for Uttar Pradesh
up_data = data[data['State/UnionTerritory'] == 'Uttar Pradesh']

# Create a graph to show the month-wise relation between confirmed cases and deaths
up_data['Month'] = up_data['Date'].dt.month
month_correlation = up_data.groupby('Month')[['Confirmed', 'Deaths']].sum().reset_index()
month_correlation['Correlation'] = month_correlation['Confirmed'].corr(month_correlation['Deaths'])
sns.scatterplot(x='Confirmed', y='Deaths', data=month_correlation, hue='Month', palette='viridis')
plt.title('Month-wise Relation between Confirmed Cases and Deaths in Uttar Pradesh')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.legend(title='Month')
plt.annotate(f'Correlation: {month_correlation["Correlation"][0]:.2f}', xy=(200000, 1500))
plt.show()
