from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

SHARE_OF_TT_SPLIT = 0.2
MODELS = [
        ('LR', LogisticRegression()),
        ('R', RidgeClassifier()),
        ('SGDC', SGDClassifier(loss='log_loss', penalty='l1')),
        ('kNN', KNeighborsClassifier(weights='distance')),
        ('SVC', SVC(C=1, gamma=0.1)),
        ('RF', RandomForestClassifier()),
        ('NB', GaussianNB())
    ]
VOTING = [
    ('SVC', SVC())
    ]