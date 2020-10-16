from phik import phik_matrix
import scipy.stats as ss
from sklearn import preprocessing

import numpy as np
import pandas as pd

# Returns an overview stats of the dataset
def get_data_stat(data):
    """
    Parameter
    data: a dataframe format of the dataset

    Returns
    dict_stats: dictionary containing all data statistics information about the data
    """
    memory = data.memory_usage().sum()/1048576 # 1Mib is equal to 1048576bytes
    missing_cells = data.isnull().sum().sum()
    duplicates = data.duplicated(subset=None, keep='first').sum()
    structure = data.shape
    dict_stats =  {"Number of variables": int(structure[1]), 
                    "Number of observations":int(structure[0]), 
                    "Missing cells": int(missing_cells), 
                    "Missing cells (%)": int(round((missing_cells * 100),2)), 
                    "Duplicate rows": int(duplicates), 
                    "Duplicate rows (%)": int(round((((duplicates)/structure[0]) * 100),2)), 
                    "Total size in memory": str(round(memory,2))+ " MiB", 
                    }
    return dict_stats

# Returns an overview stats of the prottected attributes
def get_attributes_stat(attributes, data):
    """
        Parameters
        attributes: is a list of the column names of the protected attributes
        data: a dataframe format of the dataset

        Returns
        stats_dict: dictionary containing statistical information about the columns specified
        
    """
    stats_dict = {}
    for attribute in attributes:
        
        # attributes dictionary
        attribute_dict = dict(data[attribute].value_counts())
        attribute_dict = dict(map(lambda x: (x[0], int(x[1])), attribute_dict.items()))
        if len(attribute_dict) > 5:
            final_dict = dict(list(attribute_dict.items())[:5])
            final_dict = dict(map(lambda x: (x[0], int(x[1])), final_dict.items()))
            final_dict.update({"Others": int(sum(list(attribute_dict.values())[6:]))})
        else:
            final_dict = attribute_dict

        missing = int(data[attribute].isnull().sum())
        dictionary =  {attribute: {"Distinct Count": int(data[attribute].nunique()),  
                                "Missing Count": int(missing), 
                                "Missing Percent": int(round(((missing/len(data)) * 100),2)), 
                                "Memory Size": str(data[attribute].memory_usage()/1000) + " KiB", 
                                "Attributes": final_dict}
                        }
        stats_dict.update(dictionary)
    return stats_dict

def cramers_v_test(v1,v2):    
    """ 
        Calculate Cramers V statistic for categorical-categorical association.
        Uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
        
        Source: https://doi.org/10.1016/j.jkss.2012.10.002


        Parameters:
        v1 (series) first categorical variable 
        v2 (series) second categorical variable

        Returns : (int)
        Correlation between two v1 and v2
    """

    confusion_matrix = np.array(pd.crosstab(v1,v2, rownames=None, colnames=None))
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def label_encode_categ(data):
    '''
    Performs encoding with value between 0 and n_classes-1 on all of the categorical variables of the data.

    Parameters:
    data (DataFrame) input data


    Returns: (DataFrame)
    Dataframe with all categorical data encoded.


    '''
    data = data.select_dtypes(['object'])
    
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame() 

    for i in data.columns :
        data_encoded[i]=label.fit_transform(data[i])
    
    return data_encoded

def cramers_V(data):
    '''
    Main function for cramer's V that creates matrix out of all the paired categorical variables in the data.
    Consists of two subfunctions.
    1. Get all categorical variables only (label_encode_categ)
    2. Create Cramer's V function (cramers_v_t)
    3. Create pairwise matrix of all categorical variables


    data (DataFrame) input dataframe

    Returns (DataFrame)
    Pairwise matrix of all categorical variables
    '''
    import numpy as np
    data_encoded = label_encode_categ(data)
    rows= []
    for var1 in data_encoded:
        col = []
        for var2 in data_encoded :
            cramers = cramers_v_test(data_encoded[var1], data_encoded[var2])
            col.append(cramers) 
        rows.append(col)
      
    cramers_results = np.array(rows)
    df = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)

    return df

# Returns correlation coefficients of the data
def get_corr_stat(data):
    """
    Parameters: (Dataframe)
    Data input containing all values of each variables

    Returns: (Dictionary)
    Dictionary containing the keys as the Correlation method and values as the coefficients in dataframe form
    """
    dict_corr_stat = {"Pearson's R": data.corr(method='pearson'),
                        "Spearman's P": data.corr(method='spearman'), 
                        "Kendall's T": data.corr(method='kendall'),
                        "Phi-K": data.phik_matrix(),
                        "Cramer's V": cramers_V(data)  
                        }
    return dict_corr_stat