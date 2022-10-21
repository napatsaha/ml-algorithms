# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:21:55 2022

@author: napat

Naive Bayes Categorical Classifier from scratch
"""

import numpy as np
import pandas as pd




def naive_bayes_clf(obs, data, log_prob=False, prob=False,
                    alpha=1.0):
    """
    Naive Bayes Classifier for purely categorical data.
    
    Parameters:
        obs: list, array-like
            New observation to predict.
            Vector of feature observations for a single example.
            Order of sequence must match feature column order in data.
            
        data: pandas.DataFrame
            Data frame of data to train classifier on.
            Target must be on the last column.
            
        log_prob: bool
            Whether or not to return array of log probabilities for each class.
            Default: False
            
        prob: bool
            Whether or not to return array of normalized probabilities for each class.
            Default: False
            
        alpha: float
            Smoothing parameter. Used to prevent zero counts of feature-pairs
            not observed in the dataset. 
            Will be added to all counts before normalizing.
            
    Returns:
        string
            Predicted class for the given observation
            if both log_prob and prob are False.
            
        numpy.array
            Probabilities (or log probabilities) for each class
            if either log_prob or prob are True.
    """
    
    
    
    target_name = data.columns[-1]
    
    # Calculate Likelihood: New Algorithm
    correct = data.iloc[:,:-1] == obs # Convert dataset to True/False based on conditioned values
    correct = pd.concat([correct, data[target_name]], axis=1) # Add target column back in

    # Count of Observed instances of each feature for each class
    counts = correct.groupby(target_name).sum() + alpha
    
    # Denominator: N_classes + alpha * N_categories
    class_counts = correct[target_name].value_counts() # Number of occurrences of each classes
    cat_counts = data.nunique()[:-1] # Number of categories in each feature
    
    # Turn Denominator into matrix
    denom = counts.copy() # Use counts matrix as template
    denom[:] = 0 # Set everything to zero
    denom = denom.add(class_counts, axis=0).add(alpha * cat_counts, axis=1) # Perform row-wise and column-wise addition concurrently

    # Likelihood = (count + alpha / (n_class + alpha * n_category))
    lik = counts / denom

    lik = np.log(lik).sum(axis=1) # Log transform

    # Calculate prior
    prior = data[target_name].value_counts(normalize = True)
    prior = np.log(prior) # Log transform

    # Calculate Posterior
    log_posterior = lik + prior

    
    if prob or log_prob:
        if log_prob:
            return log_posterior
        else:
            posterior = np.exp(log_posterior)
            posterior /= sum(posterior)
            return posterior
    else:
        ix_max = log_posterior.argmax()
        return log_posterior.index[ix_max]
    


if __name__ == "__main__":
    
    from sklearn import naive_bayes, preprocessing
    
    # Data for testing
    weather = pd.read_csv("weather.csv")
    feature_names = weather.columns[:-1]
    target_name = weather.columns[-1]

    obs1 = ['weekday', 'summer', 'high', 'heavy']
    obs2 = ['sunday', 'autumn', 'normal', 'slight']
    obs0 = ['weekday', 'winter', 'high', 'heavy']
    obs = [obs0, obs1, obs2]

    ## Using Created function
    pred_my = [naive_bayes_clf(o, weather) for o in obs]
    my_p = np.r_[[naive_bayes_clf(o, weather, prob=True) for o in obs]]
    my_logp = np.r_[[naive_bayes_clf(o, weather, prob=True) for o in obs]]
    
    ## Comparison with sklearn
    
    X = weather.loc[:, feature_names]
    y = weather.loc[:, target_name]
    oe = preprocessing.OrdinalEncoder()
    le = preprocessing.LabelEncoder()
    Xe = oe.fit_transform(X)
    ye = le.fit_transform(y)
    
    # Fit Data    
    nb = naive_bayes.CategoricalNB()
    nb.fit(Xe, ye)
    # tr_acc = nb.score(Xe, ye)

    # Prediction
    new_obs = pd.DataFrame(obs, columns=feature_names)
    pred_sk = le.inverse_transform(nb.predict(oe.transform(new_obs)))
    sk_p = nb.predict_proba(oe.transform(new_obs))
    sk_log_p =  nb.predict_proba(oe.transform(new_obs))

print(f"""
      COMPARISON
      ----------
      
    PREDICTION
    ----------
          
        naive_bayes_clf
        {pred_my}
        
        sklearn
        {list(pred_sk)}
        
    PROBABILITIES
    -------------
          
        naive_bayes_clf\n{my_p}
        
        sklearn\n{sk_p}
        
    LOG PROBABILITIES
    -----------------
          
        naive_bayes_clf\n{my_logp}
        
        sklearn\n{sk_log_p}
        
        
      """)