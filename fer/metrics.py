import pandas as pd
import numpy as np

def nsd(dist):
    '''
    NSD: Normalized Standard Deviation 
        - dist: numpy array with the demographic distribution of the population
    Calculates the normalized standard deviation of the distribution
    '''
    bias = np.std(dist) * len(dist) / np.sqrt(len(dist) - 1)
    return bias

def representationalBias(df, target, partition = 'train'):
    '''
    Wrapper for the NSD
        - df: a Pandas Dataframe
        - target: name of the column to analyze
        - partition: partition to filter
    Obtains the distribution of unique values of df[target], conditioned to the
    partition provided, and calculates the nsd and the number of unique values
    '''
    counts = df[df['partition']==partition][target].value_counts()
    bias = nsd(counts / counts.sum())
    unique_values = len(counts)
    return bias, unique_values


def npmi(dff, x, y, includecross=False):
    '''
    NPMI: Normalized Pointwise Mutual Information
        - dff, a Pandas Dataframe
        - x, name of one of the columns in the dataframe
        - y, name of the second column in the dataframe
        - includecross, a boolean
    Returns a Pandas Dataframe, with the columns the unique values of dff[y] and
    the rows the unique values of dff[x], and each value representing the NPMI
    score of the specific combination of values.
    The includecross boolean returns a second dataframe with the direct number of 
    occurrences for each pair of values
    '''
    with np.errstate(divide='ignore'):
        df = dff.copy()
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        cross = pd.crosstab(df[x], df[y])
        real = cross.copy()
        real[:] = cross.to_numpy() / cross.to_numpy().sum()
        expected_mat = px.dot(py)

        pmi_mat = real.to_numpy() / expected_mat
        logs = -np.log(real.to_numpy())
        with np.errstate(divide='ignore',invalid='ignore'):
            pmi_mat = np.log(pmi_mat) / logs
        pmi_mat[np.logical_and((real == 0).to_numpy(), (expected_mat > 0))] = -1
        final = real.copy()
        final[:] = pmi_mat
        if includecross:
            return final, cross
        else:
            return final

def nmi(dff, x, y):
    '''
    NMI: Normalized Mutual Information
        - dff, a Pandas Dataframe
        - x, name of one of the columns in the dataframe
        - y, name of the second column in the dataframe
    Returns a single float with the calculated NMI between the unique values
    of the provided dataframe
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        df = dff.copy()
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        real = pd.crosstab(df[x], df[y])
        real[:] = real.to_numpy() / real.to_numpy().sum()
        expected_mat = px.dot(py)
        
        pmi_mat = real.to_numpy() / expected_mat
        numerator = real.to_numpy() * np.log(pmi_mat)
        numerator[real == 0] = 0
        denominator = real.to_numpy() * np.log(real.to_numpy())
        denominator[real == 0] = 0
        final = np.sum(numerator) / (-np.sum(denominator))
        return final
    
def od(rec):
    '''
    OD: Overall Disparity
        - rec, numpy matrix with the recalls, where the rows are the classes of the
        dataset, and the columns are the demographic groups for which the recalls are
        calculated
    Returns a single float with the calculated OD for the model
    '''
    return ((1 - rec / rec.max(axis=1).to_numpy()[:, np.newaxis]).sum(axis=1) / (len(rec.columns) - 1)).mean()











