import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
titanic = pd.read_csv('Titanic.csv')

 
print(titanic.head())
print("\nMissing values in each column:")
print(titanic.isnull().sum())
print(titanic.info())

 
titanic = titanic.drop(columns=['Name'])
titanic = titanic.drop(columns=['Ticket'])

 
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

 
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("print summary of data")
print(titanic.describe()) 

 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(titanic['Age'], bins=20)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
plt.hist(titanic['Fare'], bins=20)
plt.title('Fare Distribution')

plt.show()
 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Age', y='Fare', data=titanic)
plt.title('Age vs Fare')

plt.subplot(1, 2, 2)
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Sex vs Survival')

plt.show()

 
plt.figure(figsize=(6, 5))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Pclass vs Survival')
plt.show()

 
corr_matrix = titanic.corr()
print("\nCorrelation matrix:")
print(corr_matrix)

 
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
