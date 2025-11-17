# Functions during radiomics analysis

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, make_scorer, accuracy_score, roc_auc_score, confusion_matrix

from boruta import BorutaPy

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def Boruta_select(train, test, boruta_para, boruta_file):
    '''
    Function for feature selection with Boruta
    '''
    rf = RandomForestClassifier(n_jobs = -1, max_depth = 3)

    feat_selector = BorutaPy(rf, 
                     n_estimators = boruta_para['n_estimators'], 
                     random_state = boruta_para['random_state'], 
                     alpha = boruta_para['alpha'], 
                     perc = boruta_para['perc'], 
                     max_iter = boruta_para['max_iter'], 
                     verbose = boruta_para['verbose'])

    X = train.values
    Y = test.values
    feat_selector.fit(X, Y)

    # Check the number of remaining features
    remain = feat_selector.support_

    # Export Boruta for all features
    feature_name = train.columns
    Boruta_info = pd.DataFrame([feature_name, feat_selector.ranking_])
    Boruta_info.index = ['feature_name', 'Index']
    Boruta_info.T.to_csv(boruta_file)
    
    return remain


def BestParaSearch(Xtrain, Xtest, Ytrain, Ytest):
    '''
    Function for searching the best parameters with GridSearchCV
    '''
    best_auc_index, best_auc_index_t = [], []
    best_Youden, best_Youden_t = [], []
    best_parameters = {}
    
    # GridSearch for best random state number and classifier's parameters
    for i in range(1, 500, 1):
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        parameters = {'n_estimators': range(1, 6, 1),
                      'max_depth': range(2, 5, 1),
                      'max_features': range(1, 4, 1),
                      #'min_samples_leaf': range(2, 11, 1),
                      #'min_samples_split': range(2, 12, 1),
                      'criterion': ['gini', 'entropy']
                      }
        clf = RandomForestClassifier(random_state=i)
        gs = GridSearchCV(clf,
                          parameters,
                          scoring = scoring,
                          refit = 'AUC',
                          return_train_score = True,
                          n_jobs = -1)
        gs.fit(Xtrain, Ytrain)

        y_predict_train = gs.predict_proba(Xtrain)[:, 1]
        y_predict_test = gs.predict_proba(Xtest)[:, 1]

        fpr, tpr, thresholds = roc_curve(Ytrain, y_predict_train)
        fpr_t, tpr_t, thresholds_t = roc_curve(Ytest, y_predict_test)

        best_auc_index.append(roc_auc_score(Ytrain, y_predict_train))
        best_auc_index_t.append(roc_auc_score(Ytest, y_predict_test))

        best_parameters[i] = gs.best_params_

        # Record the Youden index for next step
        Youden = tpr - fpr
        Youden_t = tpr_t - fpr_t
        best_Youden.append(Youden.max())
        best_Youden_t.append(Youden_t.max())

    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 500, 1), best_auc_index, label = 'training')
    plt.plot(range(1, 500, 1), best_auc_index_t, label = 'testing')
    plt.legend()

    # The biggest Youden index reveals the best parameters
    best_randstate = list(best_Youden_t).index(max(best_Youden_t))+1
    best_paras = best_parameters[best_randstate]

    return best_paras, best_randstate


def ROC_cutoff(data, groups):
    '''
    Function to calculate the threshold using ROC curve
    '''
    score = data/data.max()

    _death_label = np.array(list(map(int,groups)))
    label1 = (_death_label==0)*1
    label2 = (_death_label==1)*1
    
    auc1 = roc_auc_score(label1, score)
    auc2 = roc_auc_score(label2, score)

    if auc1>=auc2:
        fpr, tpr, thresholds = roc_curve(label1, data)
        auc = auc1
    else:
        fpr, tpr, thresholds = roc_curve(label2, data)
        auc = auc2
    
    ratio_list = []
    for i in range(thresholds.shape[0]):
        if tpr[i]==1:
            ratio_list.append(0)
        else:
            ratio_list.append(fpr[i]/(1-tpr[i]))

    ratio_list = np.abs(np.array(ratio_list)-1).tolist()
    thres = thresholds[ratio_list.index(np.min(ratio_list))]
    
    return thres


def KM_estimate(data, durations, groups, label1, label2, thres, ci_show=False):
    '''
    Function to perform the survival analysis
    '''
    high_durations = durations[data>=thres]
    high_groups = groups[data>=thres]
    low_durations = durations[data<thres]
    low_groups = groups[data<thres]
    
    kmf = KaplanMeierFitter()
    
    fig = plt.figure(figsize=(10,10))
    plt.xlim((0,int(np.max(durations)+5)))
    plt.xlabel('Survival time')
    plt.ylim((0,1.05))
    plt.ylabel('Survival probability')
    
    kmf.fit(high_durations, high_groups, label=label1)
    kmf.plot(ci_show=ci_show)

    kmf.fit(low_durations, low_groups, label=label2)
    kmf.plot(ci_show=ci_show)
    
    results = logrank_test(high_durations, low_durations, high_groups, low_groups)
    
    position_x = data.shape[0]*0.7
    plt.text(position_x, 0.2, 'p=%.4f' % (results.p_value))
    
    return results.p_value


def ConfidenceInterval(prediction, true_value, n_bootstraps=1000, rand_seed=42, interval=95):
    '''
    Function to conpute the confidence intervals
    '''
    bootstrapped_scores = []

    rng = np.random.RandomState(rand_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(prediction), len(prediction))
        if len(np.unique(true_value[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(true_value[indices], prediction[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
        
    # plt.hist(bootstrapped_scores, bins=50)
    # plt.title('Histogram of the bootstrapped ROC AUC scores')
    # plt.show()
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    lower = (100-interval)/200
    upper = 1-(100-interval)/200
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    
    return confidence_lower, confidence_upper

def CCC_cal(y_true, y_pred):
    '''
    Concordance correlation coefficient calculation
    '''
    # Remove NaNs
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return numerator / denominator


def ICC_cal(data, _type='ICC3'):
    '''
    Functions to calculate ICC for multiple input groups.
    Data must be a list, and each value is a dataframe with only one column (feature and values)
    '''
    length = len(data)
    feature = data[0].columns[0]
    
    data_in = data[0]
    data_in.insert(0, 'Reader', np.ones(data_in.shape[0]))
    data_in.insert(0, 'Target', range(data_in.shape[0]))
    for i in range(length-1):
        data_con = data[i+1]
        data_con.insert(0, 'Reader', np.ones(data_con.shape[0])*(i+1+1))
        data_con.insert(0, 'Target', range(data_con.shape[0]))
        
        data_in = pd.concat([data_in, data_con])
        
    icc = pg.intraclass_corr(data = data_in, 
                             targets = 'Target', 
                             raters = 'Reader', 
                             ratings = feature)
    icc_value = icc[icc['Type']==_type]['ICC'].values[0]
    icc_p = icc[icc['Type']==_type]['pval'].values[0]
    icc_CI = icc[icc['Type']==_type]['CI95%'].values[0]
    
    return icc_value, icc_p, icc_CI


def SpecificityCal(labels, pred):
    '''
    Functions to calculate Specificity for specific model.
    '''
    MCM = confusion_matrix(labels,pred)
    tn_sum = MCM[0,0] # True negative
    fp_sum = MCM[0,1] # False positive
    tp_sum = MCM[1,1] # True positive
    fn_sum = MCM[1,0] # False negative
    
    Condition_negative = tn_sum+fp_sum+1e-6  # 加上1e-6是为了防止tp_sum和fn_sum同时为0的情况
    specificity = tn_sum/Condition_negative
    return specificity


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    '''
    Functions for decision curve.
    To calculate the benefit from the model.
    '''
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model
def calculate_net_benefit_all(thresh_group, y_label):
    '''
    Functions for decision curve.
    To calculate the benefit if treating all samples.
    '''
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all