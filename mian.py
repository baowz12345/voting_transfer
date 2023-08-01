import pickle
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, recall_score, auc
from sklearn.model_selection import train_test_split

import joblib

import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics


class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, name):
        s = np.loadtxt(name, dtype=np.float32, delimiter='  ')
        end = s.shape[1] - 1
        X = s[:, :end]
        y = s[:, -1]
        train_X, text_X, train_y, text_y = train_test_split(X, y, test_size=.3, random_state=0)
        self.x_train, self.x_test, self.y_train, self.y_test = train_X, text_X, train_y, text_y

    @staticmethod
    def __Classifiers__(name=None):
        # See for reproducibility
        random_state = 100
        kernel = 1.0 * RBF(1.0)
        if name == 'Neighbors':
            return RadiusNeighborsClassifier(radius=1.0)
        if name == 'Gaussian_Process':
            return GaussianProcessClassifier(kernel=kernel, random_state=random_state)
        if name == 'Gaussian_NB':
            return GaussianNB()
        if name == 'Bernoulli_NB':
            return BernoulliNB()
        if name == 'DecisionTree':
            return tree.DecisionTreeClassifier()
        if name == 'Bagging':
            return BaggingClassifier(base_estimator=SVC())
        if name == 'RandomForest':
            return RandomForestClassifier(n_estimators=10)
        if name == 'AdaBoost':
            return AdaBoostClassifier(n_estimators=100, random_state=random_state)
        if name == 'GradientBoosting':
            return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                              random_state=random_state)
        if name == 'HistGradientBoosting':
            return HistGradientBoostingClassifier()
        if name == 'MLP':
            return MLPClassifier(random_state=random_state)

    # 1.6.2
    def __Neighbors__(self):
        # Decision Tree Classifier
        neigh = Ensemble.__Classifiers__(name='Neighbors')
        # Init Grid Search
        neigh.fit(self.x_train, self.y_train)

    # 1.7.2
    def __GPC__(self):
        # Decision Tree Classifier
        GPC = Ensemble.__Classifiers__(name='Gaussian_Process')
        # Init Grid Search
        GPC.fit(self.x_train, self.y_train)

    # 1.9.1
    def __Gaussian_NB__(self):
        # Decision Tree Classifier
        gnb = Ensemble.__Classifiers__(name='Gaussian_NB')
        # Init Grid Search
        gnb.fit(self.x_train, self.y_train)

    # 1.9.4
    def __Bernoulli_NB__(self):
        # Decision Tree Classifier
        bnb = Ensemble.__Classifiers__(name='Bernoulli_NB')
        # Init Grid Search
        bnb.fit(self.x_train, self.y_train)

    # 1.10.1
    def __DecisionTree__(self):
        # Decision Tree Classifier
        dt = Ensemble.__Classifiers__(name='DecisionTree')
        # Init Grid Search
        dt.fit(self.x_train, self.y_train)

    # 1.11.1
    def __Bagging__(self):
        # Decision Tree Classifier
        bag = Ensemble.__Classifiers__(name='Bagging')
        # Init Grid Search
        bag.fit(self.x_train, self.y_train)

    # 1.11.2
    def __RandomForest__(self):
        # Decision Tree Classifier
        Forest = Ensemble.__Classifiers__(name='RandomForest')
        # Init Grid Search
        Forest.fit(self.x_train, self.y_train)

    # 1.11.3
    def __AdaBoost__(self):
        # Decision Tree Classifier
        AdaBoost = Ensemble.__Classifiers__(name='AdaBoost')
        # Init Grid Search
        AdaBoost.fit(self.x_train, self.y_train)

    # 1.11.4
    def __GradientBoosting__(self):
        # Decision Tree Classifier
        Gdbt = Ensemble.__Classifiers__(name='GradientBoosting')
        # Init Grid Search
        Gdbt.fit(self.x_train, self.y_train)

    # 1.11.5
    def __HistGradientBoosting__(self):
        # Decision Tree Classifier
        HGdbt = Ensemble.__Classifiers__(name='HistGradientBoosting')
        # Init Grid Search
        HGdbt.fit(self.x_train, self.y_train)

    # 1.17.2
    def __MLPClassifier_1__(self):
        # Decision Tree Classifier
        MLP = Ensemble.__Classifiers__(name='MLP')
        # Init Grid Search
        MLP.fit(self.x_train, self.y_train)

    def __VotingClassifier__(self, fnameresult1, model, fnamefig):

        # Instantiate classifiers
        # Neigh = Ensemble.__Classifiers__(name='Neighbors')
        GPC = Ensemble.__Classifiers__(name='Gaussian_Process')
        gnb = Ensemble.__Classifiers__(name='Gaussian_NB')
        bnb = Ensemble.__Classifiers__(name='Bernoulli_NB')
        dt = Ensemble.__Classifiers__(name='DecisionTree')
        bag = Ensemble.__Classifiers__(name='Bagging')
        Forest = Ensemble.__Classifiers__(name='RandomForest')
        Ada = Ensemble.__Classifiers__(name='AdaBoost')
        Gdbt = Ensemble.__Classifiers__(name='GradientBoosting')
        HGdbt = Ensemble.__Classifiers__(name='HistGradientBoosting')
        MLP = Ensemble.__Classifiers__(name='MLP')
        # Voting Classifier initialization
        vc = VotingClassifier(estimators=[('Gaussian_Process', GPC), ('Gaussian_NB', gnb),
                                          ('Bernoulli_NB', bnb), ('DecisionTree', dt), ('Bagging', bag),
                                          ('RandomForest', Forest), ('AdaBoost', Ada), ('GradientBoosting', Gdbt),
                                          ('HistGradientBoosting', HGdbt), ('MLP', MLP)
                                          ], voting='soft')
        # Copy the weights of the trained voting ensemble learning model
        vc.set_params(**model.get_params())
        # Training to obtain MLP model parameters
        vc.fit(self.x_train, self.y_train)
        # Getting train and test accuracies from meta_model
        y_pred_train = vc.predict(self.x_train)
        y_pred = vc.predict(self.x_test)
        # roc curve drawing
        y_score = vc.predict_proba(self.x_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, 1 - y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        lw = 2
        # 宽度

        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('voting roc')
        plt.legend(loc="lower right")
        hun = metrics.confusion_matrix(self.y_test, y_pred)
        sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
        sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

        plt.plot(fpr, tpr,
                 lw=lw, color='red', label='voting ROC curve (area = %0.2f)' % roc_auc)
        # The false positive rate is on the abscissa, and the true rate is on the ordinate to make a curve

        with open(fnameresult1, 'a') as ex_f:
            print("************")
            print("sn = ", sn)
            print("sp = ", sp)
            print(f"Train accuracy: {accuracy_score(self.y_train, y_pred_train)}")
            print(f"Test accuracy: {accuracy_score(self.y_test, y_pred)}")
            print(f"index:{classification_report(self.y_test, y_pred)}")
            # print(y_pred_train)
            ex_f.write('sn:' + str(sn) + ' ' + 'sp:' + str(sp) + '\n')
            ex_f.writelines("Train accuracy:" + str(accuracy_score(self.y_train, y_pred_train)) + '\n')
            ex_f.writelines("Test accuracy:" + str(accuracy_score(self.y_test, y_pred)) + '\n')
            ex_f.writelines('index' + str(classification_report(self.y_test, y_pred)) + '\n')
            ##############################################
            label1 = []
            sn_clf = []
            sp_clf = []
            acc = []
            f1 = []
            recall = []

            colors = cycle(
                ['aqua', 'darkorange', 'cornflowerblue', 'green', 'navy', 'orange', 'brown', 'purple',
                 'pink', 'dimgray', 'black', 'silver'][:10])
            for clf, label, color in zip([GPC, gnb, bnb, dt, bag, Forest, Ada, Gdbt, HGdbt, MLP],
                                         ['Gaussian_Process', 'Gaussian_NB', 'Bernoulli_NB',
                                          'DecisionTree',
                                          'Bagging', 'RandomForest', 'AdaBoost', 'GradientBoosting',
                                          'HistGradientBoosting',
                                          'MLP'], colors):
                clf.fit(self.x_train, self.y_train)
                y_pred = clf.predict(self.x_test)
                y_score = clf.predict_proba(self.x_test)

                fpr, tpr, thresholds = roc_curve(self.y_test, 1 - y_score[:, 0])
                plt.plot(fpr, tpr, color=color, lw=2,
                         label=label + ' ROC curve (area = {:.2f})'.format(auc(fpr, tpr)))

                hun = metrics.confusion_matrix(self.y_test, y_pred)
                sn = format(float(hun[0, 0]) / float((hun[0, 0] + hun[0, 1])), '.3f')
                sp = format(float(hun[1, 1]) / float((hun[1, 1] + hun[1, 0])), '.3f')
                ac = accuracy_score(self.y_test, y_pred)
                f1_1 = f1_score(self.y_test, y_pred)
                recall_1 = recall_score(self.y_test, y_pred)
                sn_clf.append(sn)
                sp_clf.append(sp)
                acc.append(ac)
                f1.append(f1_1)
                recall.append(recall_1)
                label1.append(label)

            data = {'label': label1, 'ACC': acc, 'recall': recall, 'f1': f1, 'sn': sn_clf, 'sp': sp_clf}
            frame = pd.DataFrame(data)
            frame.to_csv(ex_f, index=False)
            plt.legend(loc="lower right")
            plt.savefig(fnamefig)
            plt.show()
            plt.close()
            print(frame)


if __name__ == "__main__":
    with open('model.pkl', 'rb') as f:
        model1 = pickle.load(f)
    # fname = './SeqVec1.txt'
    # fnameresult = 'test_result.txt'
    # ensemble = Ensemble()
    # ensemble.load_data(fname)
    # ensemble.__VotingClassifier__(fnameresult, model1)

    path = ".\\data\\"
    juns = ['bianxinglianquijun', 'eryanghuatan', 'jinhuangseputaoquijun', 'pilinpinyangjun', 'shuangqiganjun']
    # #juns = [ 'jinhuangseputaoquijun', 'pilinpinyangjun']
    # juns = ['eryanghuatan','shuangqiganjun']
    sites = ['binding-site_', 'active-site_']
    methods = ['SeqVec', 'TAPE', 'ProSE']
    # 'ProSE'
    for jun in juns:
        for site in sites:
            for method in methods:
                # composition path
                fname = path + jun + '/' + site + method + '.txt'
                fnameresult = path + jun + '/' + site + method + 'result' + '.txt'
                fnamefig = path + jun + '/' + site + method + 'all_ROC' + '.png'
                print(fname)
                # model
                ensemble = Ensemble()
                ensemble.load_data(fname)
                ensemble.__VotingClassifier__(fnameresult, model1, fnamefig)
                print("***********" + fname + " Finish*********")

    print("********All Over************")
