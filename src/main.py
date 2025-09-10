import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Titanic dataset from seaborn
datos_titanic = sns.load_dataset("titanic")


# Display dataset shape and first few rows
print(datos_titanic.shape)
print(datos_titanic.head())
print(datos_titanic.columns)


# Search for missing values in each column
print(datos_titanic.isna().sum())


## Fill missing values
# Replace missing 'age' values with the median age
print(datos_titanic['age'].fillna(datos_titanic['age'].median(), inplace=True))

# Replace missing 'deck' values with the most frequent deck
print(datos_titanic['deck'].fillna(datos_titanic['deck'].mode()[0], inplace=True))

# Replace missing 'embark_town' values with the most frequent embark town
print(datos_titanic['embark_town'].fillna(datos_titanic['embark_town'].mode()[0], inplace=True))


## Study the target variable
# Count the number of survivors (1) and non-survivors (0)
print(datos_titanic['survived'].value_counts())


## Visualize data distributions
# Distribution of passengers by sex
sns.countplot(x='sex', data=datos_titanic)
# Show the graph
plt.show()



# Distribution of passengers by class
sns.countplot(x='pclass', data=datos_titanic)
# Show the graph
plt.show()


# Distribution of passengers by age group ('who') 
sns.countplot(x='who', data=datos_titanic)
# Show the graph
plt.show()



## Transform string variables into numerical variables for use algorithms using dummies
# Convert categorical variables into dummy/indicator variables
datos_titanic=datos_titanic.drop(["pclass", "embarked", "alive"] , axis=1)
# Create dummy variables for categorical columns and drop the first category to avoid multicollinearity
datos_titanic_limpios = pd.get_dummies(datos_titanic, columns=["sex","embark_town","class", "who", "deck" ], drop_first=True)
# Show the rows of the cleaned dataset
print(datos_titanic_limpios.head())


## Separate the data into two sets: features (X) and target (y)
X = datos_titanic_limpios.drop("survived", axis=1)
y = datos_titanic_limpios["survived"]

## Split the data into training and test sets
from sklearn import tree
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

## Create and train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=1)

# Train the model
arbol=arbol.fit(x_train, y_train)

# Make predictions on the test set and check the accuracy
predicciones_test=arbol.predict(x_test)

# Check the accuracy of the model on the training and testing sets
print(arbol.score(x_train, y_train))
print(arbol.score(x_test, y_test))


## Create and display the confusion matrix
from sklearn.metrics import confusion_matrix
matriz_confusion = confusion_matrix(y_test, predicciones_test)
print(matriz_confusion)



## Visualize the decision tree
import graphviz

dot_data = tree.export_graphviz(arbol, out_file=None, 
                                feature_names=x_train.columns,  
                                filled=True)  
graph = graphviz.Source(dot_data, format="png")
# Save and render the graph to a file
graph.render("arbol_titanic") 
# Display the graph after creation
graph.view()  



'''
# Alternatively, use matplotlib to plot the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(arbol, feature_names=x_train.columns, filled=True, class_names=['No sobrevivió','Sobrevivió'])
plt.show()
'''

