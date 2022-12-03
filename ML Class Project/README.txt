In this folder we are going to add the scripts and results from ML class project.



####### Adding LDA

# input and output variables
X = df.drop(['type','label'],axis=1)
y = df.type
target_names = df.type.unique()

# importing the requried module
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# initializing the model with 2 components
lda = LinearDiscriminantAnalysis(n_components=2)

# fitting the dataset
X_r2 = lda.fit(X, y).transform(X)


# importing the required module
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=X_r2[:,0],
                    y=X_r2[:,1],
                    hue=df.type).set(title='LDA to visualize all attack type')
#place legend outside top right corner of plot
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
