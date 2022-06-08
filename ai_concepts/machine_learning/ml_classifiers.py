import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier


class MLClassifiers:

    def __init__(self, df_out: pd.DataFrame):
        self.df_out = df_out

    def call_ml_classifiers(self, random_state: int = 1):
        x_train, x_test, y_train, y_test = train_test_split(
            self.df_out['text'],
            self.df_out['label'],
            random_state=random_state
        )

        result_cols = ["Classifier", "Accuracy"]
        result_frame = pd.DataFrame(columns=result_cols)

        classifiers = [
            KNeighborsClassifier(10),
            SVC(),
            NuSVC(probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=100, random_state=0, bootstrap=True, class_weight=None,
                                   criterion='gini',
                                   max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0),
            AdaBoostClassifier(),
            MultinomialNB(),
            DecisionTreeClassifier(random_state=0, max_depth=2)
        ]

        for clf in classifiers:
            name = clf.__class__.__name__
            text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                 ('clf', clf), ])
            text_clf.fit(x_train, y_train)

            predicted = text_clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, predicted)
            acc_field = pd.DataFrame([[name, acc * 100]], columns=result_cols)
            result_frame = result_frame.append(acc_field)

        return result_frame
