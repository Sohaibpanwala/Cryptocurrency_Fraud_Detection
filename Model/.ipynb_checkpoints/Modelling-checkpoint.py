import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier

df = pd.read_csv('Data/dataset.csv')

df = df.drop(columns='Unnamed: 0')

features = df.iloc[:,2:]

y = df['class']


X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=42)

clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models
