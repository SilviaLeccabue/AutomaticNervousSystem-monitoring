import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt



DATA = pd.read_csv('PUPIL_DIAMETER.csv')
df_normalized = pd.read_csv('PUPIL_DIAMETER.csv')

df= DATA.iloc[:,1:-1]
df_normalized= df_normalized.iloc[:,1:-1]

# Some standardization for PCA analysis 

scaler = RobustScaler(copy=False) #or alternatively use MinMaxScaler
scaler.fit(df_normalized) 
scaler.transform(df_normalized); 

%matplotlib inline
df_normalized.boxplot()

%matplotlib inline
df.boxplot()
df.hist()

#PCA fit

# we can choose the number of components e.g. 10, the percentage of the total variance or set it to None (that means it automatically chooses the number of components)

pca = PCA(n_components=7)
pca.fit(df) 

#let's use the pca to transform the dataset

x_pca = pca.transform(df)
print("Dataset shape before PCA: ", df.shape)
print("Dataset shape after PCA: ", x_pca.shape)
pd.DataFrame(pca.explained_variance_).transpose()

#The percentage of variance explained by each of the selected components.
explained_var=pd.DataFrame(pca.explained_variance_ratio_).transpose()
explained_var

%matplotlib inline
sns.barplot( data=explained_var)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

principalDf = pd.DataFrame(data = x_pca
             , columns = ['pc1', 'pc2','pc3','pc4','pc5', 'pc6','pc7'])
principalDf.head()
principalDf['Target']=DATA['Target']

# The data in the first two PCA 

sns.scatterplot(x="pc1", y="pc2",
              hue="Target", alpha=1,
              data=principalDf);

x = df.iloc[:,0]
y = df.iloc[:,5]

df['Target'] = DATA['Target']

plt.scatter(x, y, c= df['Target'] )
plt.show()

sns.boxplot(x=df['5'])
x1 = df_normalized.iloc[:,0]
y1 = df_normalized.iloc[:,1]
x1 = df_normalized.iloc[:,0]
y1 = df_normalized.iloc[:,1]

df_normalized['Target'] = DATA['Target']

plt.scatter(x1, y1,alpha=0.8,c=df_normalized['Target'] )
plt.show()
principalKPCA = pd.DataFrame(data = zero_one_transf
             , columns = ['pc1', 'pc2'])
principalKPCA.head()


from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principalDf['pc1'], principalDf['pc2'],principalDf['pc3'], c=principalDf['Target'], s=40)
ax.view_init(0, 100)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c=DATA['Target'], s=40)
ax.view_init(0, 100)
plt.show()


# separating X and Y from the original unbalanced data frame for testing the model on different data-frames

X = df.iloc[:,1:-1]
y=df.iloc[:,-1]

X_princ = principalDf.iloc[:,0:2]
y_princ=principalDf.iloc[:,-1]
X_princ

#SPLIT DATA INTO TRAIN AND TEST SET

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_princ,y_princ,  #X_scaled
                                                    test_size =0.30, 
                                                    random_state= 123) 

print(X_train.shape)


from sklearn import svm, datasets

h = .02

# Create an instance of SVM and fit out data

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)


# Create a mesh to plot in

x_min, x_max = principalDf['pc1'].min() - 1, principalDf['pc1'].max() + 1
y_min, y_max =principalDf['pc2'].min() - 1, principalDf['pc2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
   
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_princ[:,0], X_princ[:,1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

#DEFINE THE CLASSIFIER and THE PARAMETERS GRID. CHOSEN DIFFERENT KERNEL AND GAMMA

from sklearn.svm import SVC

classifier = SVC()
parameters = {"kernel":['linear','poly'], "C":[0.1,100],"gamma":[1e-4,0.01,1]}

#IMPLEMENTED A CROSS-VALIDATION IN ORDER TO GAIN THE MAX ACCURACY AND AVOID  OVERFITTING OF A NOISY DATASET

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'accuracy', verbose=50, n_jobs=-1, refit=True)
gs = gs.fit(X_train, y_train)

print('***GRIDSEARCH RESULTS***')

print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
params = gs.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import f1_score

print('***RESULTS ON TEST SET***')
print("f1_score: ", f1_score(y_test, y_pred))
#class 0 vs class 1
df.loc[df.Target == "0", "Target"] = 0
df.loc[df.Target == "1", "Target"] = 1
df.loc[df.Target == "2", "Target"] = 2
df["Target"] = df["Target"].astype("int")
class_zero = (df.Target == 0).values
class_one = (df.Target == 1).values
class_two = (df.Target == 2).values
zero_one = df.loc[class_zero | class_one].copy()
zero_two = df.loc[class_zero | class_two].copy()
one_two = df.loc[class_one| class_two].copy()


def plot_PCA(df, transf, labels={"A":0, "B":1}):

    for label,marker,color in zip(list(labels.keys()),('x', 'o'),('blue', 'red')):

        plt.scatter(x=transf[:,0][(df.Target == labels[label]).values],
                    y=transf[:,1][(df.Target == labels[label]).values],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label)
                    )

    plt.legend()
    plt.title('PCA projection')

    plt.show()
  
from sklearn.decomposition import PCA
X = zero_one[['0','1','2','3','4','5','6']]
y = zero_one.Target.values

#PCA fit
from sklearn.decomposition import PCA

pca01 = PCA(n_components=7)
zero_one_PCA = pca01.fit(X, y)
zero_one_transf = zero_one_PCA.transform(X)
plot_PCA(zero_one, zero_one_transf, labels={"0":0, "1":1})

#let's use the pca to transform the dataset

print("Dataset shape before PCA: ", X.shape)
print("Dataset shape after PCA: ", zero_one_transf.shape)


%matplotlib inline

sns.barplot(data=explained_var_zero_one)
plt.plot(np.cumsum(pca01.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

principal_zero_one = pd.DataFrame(data = zero_one_transf
             , columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7'])
principal_zero_one.head()
principal_zero_one['Target']= zero_one.Target.values
X_princ01 = principal_zero_one.iloc[:,0:2]
y_princ01=principal_zero_one.iloc[:,-1]

#SPLIT DATA INTO TRAIN AND TEST SET of class 0 vs 1

from sklearn.model_selection import train_test_split


X_train01, X_test01, y_train01, y_test01 = train_test_split(X_princ01,y_princ01,  #X_scaled
                                                    test_size =0.1, #by default is 75%-25%
                                                    #shuffle is set True by default,
                                                    #stratify=y,
                                                    random_state= 123) #fix random seed for replicability

print(X_train01.shape)
print(X_test01.shape)


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
svc = LinearSVC()
svc.fit(X_train01, y_train01)
y_pred01 = svc.predict(X_test01)
confusion_matrix(y_test01, y_pred01)
#Accuracy
Acc=accuracy_score(y_pred=y_pred01, y_true=y_test01)
print("Accuracy:  " + str(Acc))
print("Confusion Matrix: " + str(confusion_matrix(y_test01, y_pred01)))
#class 0 vs class 2
def plot_PCA(df, transf, labels={"A":0, "B":1}):

    for label,marker,color in zip(list(labels.keys()),('x', 'o'),('blue', 'red')):

        plt.scatter(x=transf[:,0][(df.Target == labels[label]).values],
                    y=transf[:,1][(df.Target == labels[label]).values],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label)
                    )

    plt.legend()
    plt.title('PCA projection')

    plt.show()


from sklearn.decomposition import PCA
X = zero_two.drop(["Target"], axis=1).values
y = zero_two.Target.values
#PCA fit
from sklearn.decomposition import PCA

pca02 = PCA(n_components=7)
zero_two_PCA = pca02.fit(X,y) #The fit learns some quantities from the data, most importantly the "components" and "explained variance"
zero_two_transf = zero_two_PCA.transform(X)
plot_PCA(zero_two, zero_two_transf, labels={"0":0, "2":2})

#let's use the pca to transform the dataset
print("Dataset shape before PCA: ", X.shape)
print("Dataset shape after PCA: ", zero_two_transf.shape)
#VISUALIZE The percentage of variance explained by each of the selected components.
explained_var_zero_two=pd.DataFrame(pca02.explained_variance_ratio_).transpose()
explained_var
%matplotlib inline
import seaborn as sns
sns.barplot( data=explained_var_zero_two)
plt.plot(np.cumsum(pca02.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# Let see the coordinates of the data in the PCA 
#principal_zero_two = pd.DataFrame(data = zero_two_transf
#             , columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7'])
#principal_zero_two.head()

principal_zero_two = zero_two
#principal_zero_two['Target']= zero_two.Target.values
#X_princ02 = principal_zero_two.iloc[:,0:3]
#y_princ02=principal_zero_two.iloc[:,-1]

X_princ02 = principal_zero_two.iloc[:,0:7]
y_princ02=principal_zero_two.iloc[:,-1]
#SPLIT DATA INTO TRAIN AND TEST SET of class 0 vs 2

from sklearn.model_selection import train_test_split


X_train02, X_test02, y_train02, y_test02 = train_test_split(X_princ02,y_princ02,  #X_scaled
                                                    test_size =0.10, #by default is 75%-25%
                                                    #shuffle is set True by default,
                                                    #stratify=y,
                                                    random_state= 123) #fix random seed for replicability

print(X_train02.shape)
print(X_test02.shape)


svc = LinearSVC()
svc.fit(X_train02, y_train02)
y_pred02 = svc.predict(X_test02)
confusion_matrix(y_test02, y_pred02)
#Accuracy
Acc=accuracy_score(y_true=y_test02, y_pred=y_pred02)
print("Accuracy:  " + str(Acc))
print("Confusion Matrix: " + str(confusion_matrix(y_test02, y_pred02)))

#DEFINE YOUR CLASSIFIER and THE PARAMETERS GRID
from sklearn.svm import SVC

classifier = SVC()
parameters = {"kernel":['linear'], "C":[0.1,100],"gamma":[1e-4,0.01,1]}
#DEFINE YOUR GRIDSEARCH 
'''
GS perfoms an exhaustive search over specified parameter values for an estimator.
GS uses a Stratified K-Folds cross-validator
(The folds are made by preserving the percentage of samples for each class.)
If refit=True the model is retrained on the whole training set with the best found params
'''
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'accuracy', verbose=50, n_jobs=-1, refit=True)
#TRAIN YOUR CLASSIFIER
gs = gs.fit(X_train01, y_train01)
#summarize the results of your GRIDSEARCH
print('**GRIDSEARCH RESULTS**')

print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
params = gs.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#TEST ON YOUR TEST SET 
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test01)
from sklearn.metrics import f1_score
print('**RESULTS ON TEST SET**')
print("f1_score: ", f1_score(y_test01, y_pred))
Acc=accuracy_score(y_true=y_test01, y_pred=y_pred)
print("Accuracy:  " + str(Acc))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test02, y_pred))
#class 1 vs class 2
def plot_PCA(df, transf, labels={"A":0, "B":1}):

    for label,marker,color in zip(list(labels.keys()),('x', 'o'),('blue', 'red')):

        plt.scatter(x=transf[:,0][(df.Target == labels[label]).values],
                    y=transf[:,1][(df.Target == labels[label]).values],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label)
                    )

    plt.legend()
    plt.title('PCA projection')

    plt.show()


from sklearn.decomposition import PCA
X = one_two.drop(["0", "1", "2"], axis=1).values
y = one_two.Target.values

#PCA fit
from sklearn.decomposition import PCA

pca12 = PCA(n_components=7)
one_two_PCA = pca12.fit(X, y) #The fit learns some quantities from the data, most importantly the "components" and "explained variance"
one_two_transf = one_two_PCA.transform(X)
plot_PCA(one_two, one_two_transf, labels={"1":1, "2":2})

explained_var_one_two=pd.DataFrame(pca12.explained_variance_ratio_).transpose()
explained_var
%matplotlib inline
import seaborn as sns
sns.barplot( data=explained_var_one_two)
plt.plot(np.cumsum(pca12.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
df
X = df.drop(["Target"], axis=1).values
y = df[['Target']]
y.shape
#SPLIT DATA INTO TRAIN AND TEST SET 

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y,  #X_scaled
                                                    test_size =0.30, #by default is 75%-25%
                                                    #shuffle is set True by default,
                                                    #stratify=y,
                                                    random_state= 123) #fix random seed for replicability

print(X_train.shape)
print(X_test.shape)


from sklearn.svm import SVC

classifier = SVC()
parameters = {"kernel":['linear'], "C":[0.1,100],"gamma":[1e-4,0.01,1]}


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'accuracy', verbose=50, n_jobs=-1, refit=True)
#TRAIN YOUR CLASSIFIER
gs = gs.fit(X_train, y_train)

print('**GRIDSEARCH RESULTS**')

print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
params = gs.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
Acc=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy:  " + str(Acc))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
