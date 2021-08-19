import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier
import re
from tqdm import tqdm


#Building baseline models before feature selection and hyperparameter optimization


df = pd.read_csv('Data/dataset.csv')

df = df.drop(columns='Unnamed: 0')

features = df.iloc[:,2:]

y = df['class']


X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=42)


models = [LogisticRegression,LogisticRegressionCV,LinearSVC,SGDClassifier,PassiveAggressiveClassifier,RandomForestClassifier,GradientBoostingClassifier,XGBClassifier]

def score(model):
    clf = model()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test,y_pred)

accuracyscore = []
modelname = []
for model in tqdm(models):
    accuracyscore.append(score(model))
    modelname.append(str(model))
    
modelnamestr = []
for i in modelname:
    i = re.sub("<class '\w+.\w+._\w+.|'>","",i)
    modelnamestr.append(i)

modeldf = pd.DataFrame(data=[modelnamestr,accuracyscore])

modeldf = modeldf.T

#Obtained baseline model metrics
#XGBClassifier & RFClassifier base model give great accuracy

pd.DataFrame.to_csv(modeldf,'ModelMetrics.csv',index=False)