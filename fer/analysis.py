from fastai.vision.all import *
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from functools import *
import os
import random
import pathlib
import pandas as pd
import sklearn
from copy import deepcopy

from fer.datasets import *
from fer.paths import *
from fer.metrics import *

GENERAL_CMAP = 'RdPu'

def getPredsResults(results, train_name, test_name):
    r = results['predictions'][(train_name, test_name)]
    if len(r) == 2:
        y_pred = torch.argmax(r[0], 1)
        y_true = r[1]
    else:
        y_pred = r[2]
        y_true = r[1]
    return y_true, y_pred

def metricFilter(y_true, y_pred, metric, target):
    if torch.any(y_true==target):
        return metric(y_true == target, y_pred == target)
    else:
        return 0
        
def plotEmolabelStats(data, outputfile = None):
    for partition in ['train', 'val']:
        data_part = dict(map(lambda x: (x, data[x][partition]), data))
        df = pd.DataFrame(data_part)
        df['count'] = df.count(axis=1)
        df['sum'] = df.drop('count', axis=1).sum(axis=1)
        df = df.sort_values(by=['count', 'sum'], ascending=False)
        df = df.drop(['sum', 'count'], axis=1)
        df = df.T
        df = df.fillna(0)
        
#         if 'none' not in df.columns:
#             df['none'] = 0
        
#         for col in ['unknown', 'nf']:
#             df['none'] = df['none'] + df[col]
#             df = df.drop([col], axis=1)
            
#         df['others'] = 0
#         final_vocab = base_vocab + ['none', 'others']
        
#         for col in df.columns:
#             if col not in final_vocab:
#                 df['others'] = df['others'] + df[col]
#                 df = df.drop([col], axis=1)

        plt.subplots(figsize=(20,10))
        if (df > 1).any(axis=None):
            df = df.fillna(0).astype(int, errors='ignore')
            sns.heatmap(df, cmap='YlOrBr', annot=True, square=True, cbar=False, mask=df==0, fmt='d')
        else:
            sns.heatmap(df, cmap='YlOrBr', annot=True, square=True, cbar=False, mask=df==0, fmt='.1%')
        plt.yticks(rotation=0)
        plt.title(f'Raw emotion label distribution [{partition}]')

        plt.tight_layout()
        if outputfile is not None:
            outputfile = f'{FIGURES_DATA_PATH}/emotion_label.png' if outputfile == 'default' else outputfile
            plt.savefig(outputfile)
        plt.show()
    
def _getDSNamesFromResults(results):
    dsnames = list(dsname[0] for dsname in results['predictions'].keys())
    dsnames = sorted(set(dsnames), key=dsnames.index)
    return dsnames

def showResultStats(results, measure, measure_name, plot_txt=True, sorteds=True):
    names_train = results['datasets_train']
    names_eval = results['datasets_eval']
    if len(names_train) != len(names_eval):
        matrix = np.empty((len(names_train), len(names_eval)))

        for i, train_name in enumerate(names_train):
            for j, test_name in enumerate(names_eval):
                y_true, y_pred = getPredsResults(results, train_name, test_name)
                matrix[i, j] = measure(y_true, y_pred)
                
        df = pd.DataFrame(matrix, index=names_train, columns=names_eval)
        display(df)
        return df
    
    matrix = np.empty((len(names_train), len(names_eval)))
    
    supports = [int(np.sum(FERDataset2(name).size())) for name in names_eval]

    for i, train_name in enumerate(names_train):
        for j, test_name in enumerate(names_eval):
            y_true, y_pred = getPredsResults(results, train_name, test_name)
            matrix[i, j] = measure(y_true, y_pred)
    
    means_train = np.average(matrix, axis=1)
    order_train = np.argsort(means_train)[::-1]
    
    means_test = np.average(matrix, axis=0)
    order_test = np.argsort(means_test)[::-1]
    
    stats = np.stack([matrix.diagonal(), means_train, means_test, supports], axis=0).T * 100
    df = pd.DataFrame(stats, index=names_train, columns=['Self', 'Avg train', 'Avg test', 'Dataset size'])
    print(f'Full {measure_name} stats')
    print(df)
    print()
    if sorteds:
        print(f'Sorted by self {measure_name}')
        print(df.sort_values(by='Self', ascending=False))
        print()
        print(f'Sorted by {measure_name} when used for train (how good it is)')
        print(df.sort_values(by='Self', ascending=False))
        print()
        print(f'Sorted by {measure_name} when used for test (how easy it is)')
        print(df.sort_values(by='Self', ascending=False))
        print()
        
    f, ax= plt.subplots(figsize=(8,  6))

    plt.scatter(df['Dataset size'], df['Avg train'])
    plt.xlabel('Dataset size')
    plt.ylabel(f'Average {measure_name} when used for train')
    plt.xscale('log')

    for txt in df.index:
        ax.text(df.loc[txt, 'Dataset size'], df.loc[txt, 'Avg train'],
                cleanDatasetName(txt), ha="left", va="bottom", rotation=0,
                bbox=dict(boxstyle="Round4", facecolor='lightgray',alpha=0.6))

    plt.show()
        
    return df
    
def plotResultMatrix(results, 
                     measure, 
                     measure_name, 
                     plot_txt=True, 
                     sort='average_keeppartials', 
                     outputfile = None, 
                     cmap='Blues'):
    names_train = results['datasets_train']
    names_eval = results['datasets_eval']
    names_composed = []
    if 'datasets_composed' in results:
        names_composed = results['datasets_composed']
    matrix = np.empty((len(names_train), len(names_eval) + len(names_composed)))

    for i, train_name in enumerate(names_train):
        for j, test_name in enumerate(names_eval + names_composed):
            y_true, y_pred = getPredsResults(results, train_name, test_name)
            matrix[i, j] = measure(y_true, y_pred)
    
    if ((sort == 'average' or sort == 'average_keeppartials') and
        (';'.join(sorted(names_train)) == ';'.join(sorted(names_eval)))):
        # This only works for square experiments (names_train == names_eval)
        means = np.average(matrix, axis=1) - (matrix.diagonal() / len(matrix))
        
        if sort == 'average_keeppartials':
            excludeFromSort = list()
            partials = list(map(lambda x: '#' in x, names_train))
            means[partials] = np.arange(len(partials))[partials] - 100
        
        order = np.argsort(means)[::-1]

        matrix[np.arange(len(order)), :] = matrix[order, :]
        matrix[:, np.arange(len(order))] = matrix[:, order]
        names_train = [names_train[o] for o in order]
        names_eval = [names_eval[o] for o in order]
    
    print(80 * '-')
    print(f'Cross evaluation, {measure_name}')
    print(f'Diagonal mean: {np.mean(matrix.diagonal())}')
    print(f'Global mean: {np.mean(matrix)}')
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(f'Cross evaluation, {measure_name}')
    tick_marks_y = np.arange(len(names_train))
    plt.yticks(tick_marks_y, names_train, rotation=0)
    tick_marks_x = np.arange(len(names_eval) + len(names_composed))
    plt.xticks(tick_marks_x, names_eval + names_composed, rotation=90)

    if plot_txt:
        thresh = (matrix.max() + matrix.min()) / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            coeff = f'{matrix[i, j]*100:.0f}%'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('Train')
    plt.xlabel('Test')
    plt.grid(False)
#     plt.tight_layout(rect=(0, 0, 1, 1))
    
#     if outputfile is not None:
#             savepath = f'{RESULTS_PATH}/{results["exp"]}.png' if outputfile == 'default' else outputfile
#             plt.savefig(savepath, bbox_inches="tight")
    
    plt.show()
    
def plotConfusionMatrixes(results, train_names=None, test_names=None):
    train_names = train_names if train_names is not None else results['datasets_eval'] 
    test_names = test_names if test_names is not None else results['datasets_eval'] 
    
    vocab = results['vocab']

    for i, train_name in enumerate(train_names):
        for j, test_name in enumerate(test_names):
            y_true, y_pred = getPredsResults(results, train_name, test_name)
            
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=range(len(vocab)))
            disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=vocab)
            disp.plot(cmap='Blues')
            plt.title(f'Confusion matrix for {train_name} tested on {test_name}')
            plt.show()
    
def printClassificationReports(results, vocab=base_vocab):
    names = results['datasets_eval']
    
    for i, train_name in enumerate(names):
        for j, test_name in enumerate(names):
            y_true, y_pred = getPredsResults(results, train_name, test_name)
            
            print(f'--------')
            print(f'Classification report for {train_name} tested on {test_name}')
            print(sklearn.metrics.classification_report(y_true, y_pred, target_names=vocab))
            
            
def showResultExamples(results, n=16, sort='largest-loss', train_names=None, test_names=None):
    train_names = train_names if train_names is not None else results['datasets_eval'] 
    test_names = test_names if test_names is not None else results['datasets_eval']
        
    vocab = results['vocab']
    
    for train_name in train_names:
        for test_name in test_names:
            ds = FERDataset2(test_name, load=False)
            
            fig, axs = plt.subplots(int(np.ceil(np.sqrt(n))), 
                                    int(np.ceil(np.sqrt(n))),
                                    figsize=(10, 10))
            axs = axs.ravel()
    
            res = results['predictions'][(train_name, test_name)]
            res = list(zip(*res))
            if sort == 'random':
                chosen = random.choices(res, k=n)
            elif sort == 'largest-loss':
                res.sort(key=lambda x: 0 if x[1] == x[2] else torch.max(x[3]))
                chosen = res[-n:]
                
            for i, pred in enumerate(chosen):
                axs[i].imshow(Image.open(ds.filepath(pred[0], absolute=True)))
                title = axs[i].set_title(f'Label: {vocab[pred[1]]}, \nPredicted: {vocab[pred[2]]}, \nConfidence: {torch.max(pred[3]):.2f}')
                axs[i].axis('off')
                if pred[1] == pred[2]:
                    plt.setp(title, color='g') 
                else:
                    plt.setp(title, color='r') 
                    
            plt.suptitle(f'Examples of {test_name} as evaluated by {train_name}')
            plt.tight_layout()
            plt.show()

def clean_mat(df):
    dffilt = df[df["label"].isin(base_vocab)]
    dffilt = dffilt.replace({'0-2':'00-02', '3-9':'03-09'})
    return dffilt

# def getDemographyStats(names, partition='both', conf_thr=0, identity_aggregated=True, normalize=True, targets=["age", "race", "gender"]):
#     demographic_stats = {}
    
#     for name in names:
#         ds = FERDataset2(name, load=False)
#         ddata = clean_mat(ds.getAggregatedCSV(identity_aggregated=identity_aggregated, conf_thr=conf_thr))
#         if partition != 'both':
#             ddata = ddata[ddata['partition'] == partition]
#         stats = {}
#         for target in targets:
#             tstats = ddata.loc[ddata[target] != 'Unknown', target].value_counts(sort=False, normalize=normalize).sort_index()
#             stats[target] = tstats

#         demographic_stats[name] = stats
        
#     return demographic_stats

def getDemographyStats(names, partition='both', conf_thr=0, identity_aggregated=True, normalize=True, targets=["age", "race", "gender"]):
    demographic_stats = {}
    
    for name in names:
        ds = FERDataset2(name, load=False)
        ddata = clean_mat(ds.getAggregatedCSV(identity_aggregated=identity_aggregated, conf_thr=conf_thr))
        if partition != 'both':
            ddata = ddata[ddata['partition'] == partition]
        stats = {}
        for target in targets:
            tstats = ddata.loc[ddata[target] != 'Unknown', target].value_counts(sort=False, normalize=normalize).sort_index()
            stats[target] = tstats

        demographic_stats[name] = stats
        
    return demographic_stats

def plotDemographicStats(data, targets=['race', 'gender', 'age'], outputfile = None):
    ret = {}
    for target in targets:
        tdata = {}
        for dsname, stats in data.items():
            tdata[dsname] = stats[target]

        df = pd.DataFrame(tdata)
        
        df.rename(columns = {'oulu-casia-recrop':'oulu-casia'}, inplace = True)

        rows = df.shape[0]
        cols = df.shape[1]
        size = 0.8
        plt.subplots(figsize=(size*(cols + 1), size*(rows + 2)))
        if np.any(df.to_numpy() > 1):
            sns.heatmap(df, cmap='Blues', annot=True, square=True, cbar=False, fmt='.0f')
        else:
#             sns.heatmap(df, cmap='Blues', annot=True, square=True, cbar=False, fmt='.1%')
            sns.heatmap(df, cmap='Blues', annot=True, square=True, cbar=False, fmt='.1%')
        plt.yticks(rotation=0)
        plt.title(f'Recognized {target} distribution'.capitalize())
        
#         plt.tight_layout()
        if outputfile is not None:
            savepath = f'{FIGURES_DATA_PATH}/demographic_{target}.pgf' if outputfile == 'default' else outputfile
            plt.savefig(savepath)
        plt.show()
        
        ret[target] = df
        
    return ret

def getCorrelationsLabelsToDemography(names, stat='npmi', conf_thr=0, identity_aggregated=True, targets = ["age", "race", "gender"]):
    
    metrics = {}
    supports = {}

    for target in targets:
        metrics[target] = {}
        supports[target] = {}
    
    for name in names:
        ds = FERDataset2(name, load=False)
        df = clean_mat(ds.getAggregatedCSV(identity_aggregated=identity_aggregated, conf_thr=conf_thr))

        for target in targets:
            corr = pd.crosstab(df["label"], df[target])
            if 'Unknown' in corr.columns:
                corr = corr.drop('Unknown', axis=1)
            corr_mat = corr.to_numpy()
            supports[target][name] = corr.copy()
            support_target = corr.sum(axis=0)
            relative_target = corr.copy()
            relative_target[:] = corr_mat / support_target.to_numpy()[np.newaxis, :]
            support_label = corr.sum(axis=1)
            relative_label = corr.copy()
            relative_label[:] = corr_mat / support_label.to_numpy()[:, np.newaxis]
            pmi_mat = pmi(df, 'label', target)

            computed = None

            if stat == 'absolute':
                computed = corr
            elif stat == 'relative-label':
                computed = relative_label
            elif stat == 'relative-target':
                computed = relative_target
            elif stat == 'adjusted-target':
                computed = relative_target
                avg = (corr_mat.sum(axis=1) / corr_mat.sum())[:, np.newaxis]
                computed[:] = computed - avg
            elif stat == 'pmi':
                computed = pmi_mat
            elif stat == 'npmi':
                computed = npmi(df, 'label', target)
            elif stat == 'lmi':
                computed = lmi(df, 'label', target)

            if 'Unknown' in computed.columns:
                computed = computed.drop('Unknown', axis=1)
                
            metrics[target][name] = computed
            
            for i, col in enumerate(corr.columns):
                for j, label in enumerate(corr.index):
                    df_filt = df[df[target] == col]
                    df_filt = df_filt[df_filt['label'] == label]
                    cont = df_filt['sequence'].nunique()
                    supports[target][name].at[label, col] = cont
            
    return metrics, supports
    
def aggregateCorrelationsLabelsToDemography(names, metrics, supports, support_thr=0, datasets_thr=0):
    means = {}
    stds = {}
    counts = {}
    
    target_values = {}
    label_values = None
    
    label_supports = {}
    target_supports = {}
    
    for target in metrics:
        target_values_i = [m[1].columns for m in  metrics[target].items()]
        target_values_i.sort(key=lambda x: len(x))
        target_values_i = list(target_values_i[-1])
        target_values[target] = target_values_i

        label_values = [m[1].index for m in  metrics[target].items()]
        label_values.sort(key=lambda x: len(x))
        label_values = list(label_values[-1])

        dataset_values = names
        
#         label_supports[target] = np.zeros((len(dataset_values), len(label_values)), dtype=np.int)
#         target_supports[target] = np.zeros((len(dataset_values), len(target_values_i)), dtype=np.int)
        
        final_supports = np.zeros((len(dataset_values), len(label_values), len(target_values_i)), dtype=np.int)

        computed = np.zeros((len(dataset_values), len(label_values), len(target_values_i)))
        computed[:] = np.nan
        
        for di, d in enumerate(dataset_values):
            m = metrics[target][d]
            s = supports[target][d]
            ls = supports[target][d].sum(axis=1)
            ts = supports[target][d].sum(axis=0)
            for ti, t in enumerate(target_values_i):
                for li, l in enumerate(label_values):
                    try:
                        if (ls[l] >= support_thr) and (ts[t] >= support_thr):
                            computed[di, li, ti] = m.at[l, t]
                            final_supports[di, li, ti] = int(s.at[l, t])
                    except KeyError:
                        pass
                        
        counts[target] = np.count_nonzero(~np.isnan(computed), axis=0)
        
        computed[:, counts[target] < datasets_thr] = np.NaN
        final_supports[:, counts[target] < datasets_thr] = 0
            
        label_supports[target] = np.sum(final_supports, axis=(0, 2))
        target_supports[target] = np.sum(final_supports, axis=(0, 1))
 
        means[target] = np.nanmean(computed, axis=0)
        stds[target] = np.nanstd(computed, axis=0)
        
    return means, stds, counts, label_values, target_values, label_supports, target_supports

def plotCorrelationsLabelsToDemography(means, stds, counts, label_values, target_values, label_supports, target_supports, stat='npmi',  outputfile = None):
    for target in means:
        target_values_i = target_values[target]
        mean = means[target]
        std = stds[target]
        count = counts[target]

        annotation = pd.DataFrame(mean)
        annotation2 = pd.DataFrame(std)
        if np.all(std == 0):
            for c in annotation.columns:
                annotation[c] = annotation[c].apply(lambda x: f'${x:.2f}$')
        else:
            for c in annotation.columns:
                annotation[c] = annotation[c].apply(lambda x: f'${x:.2f}$') + '\n$\pm' + annotation2[c].apply(lambda x: f'{x:.2f}$')
        
        if stat == 'npmi':
            title = 'NPMI'
            fmt = 's'
            center=0.00
            cmap = 'RdBu'
            vmin = -0.2
            vmax = 0.2
        
        elif stat == 'lmi':
            title = 'LMI'
            fmt = 's'
            center=0.00
            cmap = 'RdBu'
            vmin = -0.01
            vmax = 0.01
        
        boxsize = 0.8
        
        
        rows = int(len(label_values))
        cols = int(len(target_values_i))
        f, ax = plt.subplots(2, 3, figsize=(boxsize*(cols + 1.5), boxsize*(rows+1)), sharex='col', sharey='row',
                             gridspec_kw={'width_ratios': [1, cols, 0.5],
                                          'height_ratios': [rows, 1]})
        cbarax = f.add_subplot(233)

        sns.heatmap(label_supports[target][:, np.newaxis], ax = ax[0, 0],
                    cmap='Blues', annot=True, fmt='d',
                    square=True, linewidths=.5, cbar=False, vmin=0)
        sns.heatmap(target_supports[target][np.newaxis, :], ax = ax[1, 1],
                    cmap='Blues', annot=True, fmt='d',
                    square=True, linewidths=.5, cbar=False, vmin=0)
        sns.heatmap(mean, ax = ax[0, 1],
                    cmap=cmap , annot=annotation, fmt=fmt,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, center=center,
                    cbar_ax=cbarax, vmin=vmin, vmax=vmax)

        ax[0, 0].tick_params(bottom=False)
        ax[1, 1].tick_params(left=False)
        ax[0, 1].tick_params(left=False, bottom=False)

        ax[0, 0].set_yticklabels(labels=label_values)
        ax[1, 1].set_xticklabels(labels=target_values_i, rotation=45)

        ax[0, 1].set_xlabel('')
        ax[0, 1].set_ylabel('')

        ax[0, 0].set_ylabel('label')
        ax[1, 1].set_xlabel(target)

        ax[1,0].axis('off')
        ax[0,2].axis('off')
        ax[1,2].axis('off')

        plt.suptitle(title)
        
        if outputfile is not None:
            savepath = f'{FIGURES_DATA_PATH}/{stat}_{outputfile}_{target}.pgf'
            plt.savefig(savepath)
        
        plt.show()

def getMetricLabelsToDemography(results, 
                                train_names, test_names, 
                                metric_tuple=(sklearn.metrics.accuracy_score, 'accuracy'),
                                correct_by_label=False,
                                aggregate_labels='none',
                                targets=["age", "race", "gender"],
                                vocab= base_vocab):
    metric, metric_name = metric_tuple
    
    calculated_metrics = {}
    for train in train_names:
        calculated_metrics[train] = {}
        
        for test in test_names:
            ds = FERDataset2(test, vocab=vocab, load=False)
            df = ds.getAggregatedCSV()
            df = df[df['partition'] == 'val']

            res = pd.DataFrame(results['predictions'][(train, test)], 
                               index=['img_path', 'y_true', 'y_pred', 'preds']).T
#             res['img_path'] = res['img_path'].apply(ds.filepath)
#             df['img_path'] = df['img_path'].apply(ds.filepath)
            
            res = res.set_index('img_path', drop=False)
            res.index.names = ['index']
            df = df.set_index('img_path', drop=False)
            df.index.names = ['index']
            merged = res.join(df, lsuffix='source')

            # Alternative if index are not compatible
#             merged = res.merge(df, on='img_path')
            merged['y_true'] = merged['y_true'].apply(lambda x: x.numpy())
            merged['y_pred'] = merged['y_pred'].apply(lambda x: x.numpy())
            
#             merged['y_pred'] = np.random.randint(7, size=len(merged))

            metrics_inners = {}
    
            for target in targets:
#                 reference_metric = {}

#                 for emo_label in merged['label'].unique():
#                     if correct_by_label:
#                         filtered2 = merged[merged['label'] == emo_label]
#                         y_true = np.array(filtered2['y_true'])
#                         y_pred = np.array(filtered2['y_pred'])
#                         m = metric(y_true, y_pred)
#                     else:
#                         m = 0
#                     reference_metric[emo_label] = m
            
                metrics_inner = {}

                for target_label in merged[target].unique():
                    filtered = merged[merged[target] == target_label]
                    metrics_inner[target_label] = {}
                    
                    if aggregate_labels == 'none':
                        for emo_label in filtered['label'].unique():
                            filtered2 = filtered[filtered['label'] == emo_label]
                            y_true = np.array(filtered2['y_true'])
                            y_pred = np.array(filtered2['y_pred'])
                            m = metric(y_true, y_pred)
                            metrics_inner[target_label][emo_label] = m
                    elif aggregate_labels == 'macro':
                        aux = []
                        for emo_label in filtered['label'].unique():
                            filtered2 = filtered[filtered['label'] == emo_label]
                            y_true = np.array(filtered2['y_true'])
                            y_pred = np.array(filtered2['y_pred'])
                            aux.append(metric(y_true, y_pred))
                        metrics_inner[target_label][emo_label] = np.mean(aux)
                    elif aggregate_labels == 'micro':
                        y_true = np.array(filtered['y_true'])
                        y_pred = np.array(filtered['y_pred'])
                        m = metric(y_true, y_pred)
                        metrics_inner[target_label]['micro'] = m

                metrics_inner = pd.DataFrame(metrics_inner).T
                if correct_by_label:
                    metrics_inner = metrics_inner - metrics_inner.mean(axis=0)
#                 metrics_inner.assign(sum=df.sum(axis=1)) \
#                     .sort_values(by='sum', ascending=False) \
#                     .iloc[:, :-1]
                metrics_inner = metrics_inner.fillna(0)

                metrics_inner = metrics_inner.sort_index()
                metrics_inner = metrics_inner.reindex(sorted(metrics_inner.columns), axis=1)

                metrics_inners[target] = metrics_inner.T

            calculated_metrics[train][test] = metrics_inners 
            
    return calculated_metrics

def aggregateMetricLabelsToDemography(calculated_metrics, train_names, test_names):
    means = {}
    stds = {}
    targets = calculated_metrics[train_names[0]][test_names[0]].keys()
    
    emo_labels = base_vocab
    target_labels = {}
    
    for target in targets:
        target_labels[target] = set()
        for train in train_names:
            for test in test_names:
                target_labels[target] = target_labels[target].union(set(calculated_metrics[train][test][target].columns))
        target_labels[target] = list(target_labels[target])
        target_labels[target].sort()
                
    for target in targets:
        values = np.full((len(train_names), len(test_names), len(emo_labels), len(target_labels[target])), np.NaN)
        for tr, train in enumerate(train_names):
            for te, test in enumerate(test_names):
                m = calculated_metrics[train][test][target]
                for e, emo_label in enumerate(emo_labels):
                    for t, target_label in enumerate(target_labels[target]):
                        if emo_label in m.index and target_label in m.columns:
                            values[tr, te, e, t] = m.loc[emo_label, target_label]
        means[target] = np.nanmean(values, axis=(0, 1))
        stds[target] = np.nanstd(values, axis=(0, 1))
    
        means[target] = pd.DataFrame(means[target], index=emo_labels, columns=target_labels[target])
        stds[target] = pd.DataFrame(stds[target], index=emo_labels, columns=target_labels[target])
    
    return means, stds


def plotMetricLabelsToDemography(means, stds):
    
    widths = []
    targets = means.keys()
    
    for target in targets:
        widths.append(len(means[target].columns))

    rows = len(base_vocab)
    cols = np.sum(widths)
    size = 1
    fig, axs = plt.subplots(1, len(widths), figsize=(size * (cols + 1*len(widths)), size * (rows + 2)),
                            gridspec_kw={'width_ratios': widths,
                                         'height_ratios': [1]})
    if len(widths) == 1:
        axs = [axs]
    for i, target in enumerate(targets):
        annotation = pd.DataFrame(means[target]).copy()
        annotation2 = pd.DataFrame(stds[target]).copy()
        if np.all(stds[target] == 0):
            for c in annotation.columns:
                annotation[c] = annotation[c].apply(lambda x: f'${x:.2f}$')
        else:
            for c in annotation.columns:
                annotation[c] = annotation[c].apply(lambda x: f'${x:.2f}$') + '\n$\pm' + annotation2[c].apply(lambda x: f'{x:.2f}$')
        
        sns.heatmap(means[target],
                    cmap='RdBu' , annot=annotation, fmt='s',
                    square=True, linewidths=.5, cbar=False, ax=axs[i],
                    vmin = -0.2, vmax = 0.2, center = 0)

    plt.show()


def plotMetricForDemography(results, train_names, test_names, metric_tuple, targets=['age', 'race', 'gender']):
    metric, metric_name = metric_tuple
    
    for test in test_names:
        print(f'For test dataset [{test}]')
        
        ds = FERDataset2(test, load=False)
        df = ds.getAggregatedCSV()
        df = df[df['partition'] == 'val']

        metrics = {}
        target_labels = {}
        widths = []
        
        for target in ["race", "gender"]:
                metrics[target] = {}
        
        for train in train_names:
            res = pd.DataFrame(results['predictions'][(train, test)], 
                               index=['img_path', 'y_true', 'y_pred', 'preds']).T
            res['img_path'] = res['img_path'].apply(ds.filepath)
            res.set_index('img_path')
            df['img_path'] = df['img_path'].apply(ds.filepath)
            df.set_index('img_path')

            merged = res.merge(df)
            
            target_labels = {}
            widths = []

            for target in targets:
                unique_targets = merged[target].unique()
                metrics[target][train] = {}
                target_labels[target] = unique_targets
                for i, target_label in enumerate(unique_targets):
                    filtered = merged[merged[target] == target_label]
                    y_true = filtered['y_true'].apply(lambda x: x.numpy())
                    y_pred = filtered['y_pred'].apply(lambda x: x.numpy())
                    m = metric(y_true, y_pred)
                    support = len(filtered)
                    metrics[target][train][f'{target}.{target_label}'] = m

                widths.append(len(unique_targets))

        for target in targets:
            rows = len(metrics[target])
            cols = len(next(iter(metrics[target].values())))
            size = 1
            fig, axs = plt.subplots(1, 1, figsize=(size * (cols + 1.5), size * (rows + 1.5)))

            metricsdf = pd.DataFrame(metrics[target]).T
            metricsdf = metricsdf - metricsdf.to_numpy().mean(axis=1, keepdims=True)
            metricsdf = metricsdf.sort_index()
            metricsdf = metricsdf.reindex(sorted(metricsdf.columns), axis=1)
            sns.heatmap(metricsdf, cmap='Blues', annot=True, fmt='.2%')
            plt.show()

def cleanDatasetName(n):
    n = n.replace('-recrop', '')
    n = n.upper()
    if n == 'FERPLUS': return 'FERPlus'
    elif n == 'OULU-CASIA': return 'Oulu-CASIA'
    else: return n


def datasetProfile(path_or_df, target, vocab=None):
    if type(path_or_df) == str:
        try:
            df = FERDataset2(path_or_df, load=False).df
        except:
            print(f'Read dataset from [{path_or_df}]')
            df = pd.read_csv(path_or_df, index_col=False)
    else:
        print(f'Given dataset')
        df = path_or_df
        
    if vocab is not None:
        df = df[df['label'].isin(vocab)]
        
    train_l = (df.partition == 'train').sum()
    train_v = (df.partition == 'val').sum()
    print(f'Train dataset size: {train_l}, validation size: {train_v}')
    
    df_train = df[df['partition'] == 'train']
    df_val = df[df['partition'] == 'val']

    train_count = pd.DataFrame(df_train['dataset_name'].value_counts()).rename(columns = {'dataset_name':'train'})
    val_count = pd.DataFrame(df_val['dataset_name'].value_counts()).rename(columns = {'dataset_name':'val'})
    cat = pd.concat((train_count, val_count), axis=1)
    cat.plot(kind='bar')
    plt.title('Source dataset distribution')
    plt.show()

    train_count = pd.DataFrame(df_train[target].value_counts()).rename(columns = {target:'train'})
    val_count = pd.DataFrame(df_val[target].value_counts()).rename(columns = {target:'val'})
    cat = pd.concat((train_count, val_count), axis=1)
    cat.plot(kind='bar')
    plt.title(f'{target.capitalize()} distribution')
    plt.show()

    train_count = pd.DataFrame(df_train['label'].value_counts()).rename(columns = {'label':'train'})
    val_count = pd.DataFrame(df_val['label'].value_counts()).rename(columns = {'label':'val'})
    cat = pd.concat((train_count, val_count), axis=1)
    cat.plot(kind='bar')
    plt.title(f'Label distribution')
    plt.show()
    
    for partition in ['train', 'val']:
        display(f'NPMI analysis for {partition} partition')
        df_partition = df[df['partition'] == partition]

        cross = pd.crosstab(df_partition['label'], df[target])
        display(cross)

        n = npmi(df_partition, 'label', target)
        plotNPMIMatrix(n)
        plt.show()

def extractRepetitionResult(results, n):
    results_copy = deepcopy(results)
    predictions = {}
    typeReps = len(next(iter(results_copy['predictions'].keys()))) # 2 or 3
    if typeReps == 3:
        for train, test, it in results_copy['predictions']:
            if it == n:
                predictions[(train, test)] = results_copy['predictions'][(train, test, it)]
    elif typeReps == 2:
        for train, test in results_copy['predictions']:
            predictions[(train, test)] = results_copy['predictions'][(train, test)][n]
    results_copy['predictions'] = predictions
    return results_copy

def aggregateRepetitionResults(results):
    results_copy = deepcopy(results)
    predictions = {}
    typeReps = len(next(iter(results_copy['predictions'].keys()))) # 2 or 3
    if typeReps == 3:
        for train, test, it in results_copy['predictions']:
            current = results_copy['predictions'][(train, test, it)]
            if (train, test) not in predictions:
                predictions[(train, test)] = []
                for i in range(len(current)):
                    predictions[(train, test)].append(current[i])
            elif it == 2:
                for i in range(len(current)):
                    if type(predictions[(train, test)][i]) == list:
                        predictions[(train, test)][i] += current[i]
                    else:
                        predictions[(train, test)][i] = torch.cat([predictions[(train, test)][i], current[i]])
    elif typeReps == 2:
        for train, test in results_copy['predictions']:
            for current in results_copy['predictions'][(train, test)]:
                if (train, test) not in predictions:
                    predictions[(train, test)] = []
                    for i in range(len(current)):
                        predictions[(train, test)].append(current[i])
                else:
                    for i in range(len(current)):
                        if type(predictions[(train, test)][i]) == list:
                            predictions[(train, test)][i] += current[i]
                        else:
                            predictions[(train, test)][i] = torch.cat([predictions[(train, test)][i], current[i]])
    results_copy['predictions'] = predictions
    return results_copy

def privilege_measurement(accuracies):
    # Receives a list of matrixes, each one with columns(demography) and rows(labels),
    # calculates de metric and returns mean and std
    m = [od(acc) for acc in accuracies]
    return np.mean(m), np.std(m)

def bias_measurement(accuracies):
    # Receives a list of matrixes, each one with columns(demography) and rows(labels),
    # calculates de metric and returns mean and std
    m = [(1 - acc / acc.max(axis=1).to_numpy()[:, np.newaxis]).max(axis=1).mean() for acc in accuracies]
    return np.mean(m), np.std(m)

def balancedaccuracy(results, train, test, balancetarget):
    ds = FERDataset2(test, load=False)
    df = ds.getAggregatedCSV(identity_aggregated=False)
    df = df[df['partition'] == 'val']
    
    res = pd.DataFrame(results['predictions'][(train, test)], 
                               index=['img_path', 'y_true', 'y_pred', 'preds']).T

    res['img_path'] = res['img_path'].apply(ds.filepath)
    df['img_path'] = df['img_path'].apply(ds.filepath)
    
    res = res.set_index('img_path', drop=False)
    res.index.names = ['index']
    df = df.set_index('img_path', drop=False)
    df.index.names = ['index']
    merged = res.join(df, lsuffix='source')

    # Alternative if index are not compatible
    merged['y_true'] = merged['y_true'].apply(lambda x: x.numpy())
    merged['y_pred'] = merged['y_pred'].apply(lambda x: x.numpy())
    
    acc = []
    for target in merged[balancetarget].unique():
        filtered = merged[merged[balancetarget] == target]
        acc = sklearn.metrics.accuracy_score(filtered['y_true'], filtered['y_pred'])
    return(np.mean(acc))

def bias_analysis(results, train_names, test_names, targets=['age', 'race', 'gender'], measure=bias_measurement, vocab=base_vocab):
    metric, metric_name = (sklearn.metrics.accuracy_score, 'accuracy')
    calc_metrics = {}
    summaries = {}
    for test_name in test_names:
        calculated_metrics = [getMetricLabelsToDemography(extractRepetitionResult(results, i), 
                                                     train_names=train_names, 
                                                     test_names=[test_name], 
                                                     metric_tuple=(metric, metric_name),
                                                     correct_by_label=False,
                                                     targets=targets,
                                                     vocab=vocab) for i in range(results['reps'])]
        calc_metrics[test_name] = calculated_metrics
        
        summaries[test_name] = {}
        
        for target in targets:
            display(f'Metrics for target demographic [{target}], test dataset [{test_name}]')
            summary = {}
            for train_name in train_names:
                if '_sub_' in train_name:
                    base_name = train_name.split('_sub_')[0]
                    portion = float(train_name.split('_sub_')[1])
                    ds = FERDataset2(base_name, load=False)
                    size = int(ds.size()[0] * portion)
                else:
                    ds = FERDataset2(train_name, load=False)
                    size = ds.size()[0]

                specifics = [calculated_metrics[i][train_name][test_name][target]  for i in range(results['reps'])]
                bias_mean, bias_std = measure(specifics)
                accuracies = [sklearn.metrics.accuracy_score(*getPredsResults(extractRepetitionResult(results, i), train_name, test_name)) 
                              for i in range(results['reps'])]
                summary[train_name] = {
                    'size': size,
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'metric_mean': bias_mean,
                    'metric_std': bias_std,
                }
            display(pd.DataFrame(summary).T.round(2).style.format('{:.2f}').background_gradient(cmap='RdPu'))
            summaries[test_name][target] = pd.DataFrame(summary).T
            
    return summaries, calc_metrics

def pmi(dff, x, y):
    with np.errstate(divide='ignore'):
        df = dff.copy()
        df1 = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        df1 = df1 / df1.sum()
        df2 = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        df2 = df2 / df2.sum()
        real = pd.crosstab(df[x], df[y])
        real[:] = real.to_numpy() / real.to_numpy().sum()
        expected_mat = df1.dot(df2)

        pmi_mat = real.to_numpy() / expected_mat
        pmi_mat = np.log(pmi_mat)
        final = real.copy()
        final[:] = pmi_mat
        return final 

def lmi(dff, x, y):
    with np.errstate(divide='ignore'):
        df = dff.copy()
        df1 = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        df1 = df1 / df1.sum()
        df2 = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        df2 = df2 / df2.sum()
        real = pd.crosstab(df[x], df[y])
        real[:] = real.to_numpy() / real.to_numpy().sum()
        expected_mat = df1.dot(df2)

        pmi_mat = real.to_numpy() / expected_mat
        lmi_mat = np.log(pmi_mat) * real
        final = real.copy()
        final[:] = lmi_mat
        return final 
    

def plotMatrix(matrix, **kwargs):
    defkwargs = {
        'annot': True,
        'fmt': '.2f',
        'square': True,
        'cmap': 'RdBu',
        'cbar_kws': {"shrink": 0.5}
    }
    if 'ax' not in kwargs:
        rows = len(matrix)
        cols = len(matrix.iloc[0])
        size = 0.6
        fig, ax = plt.subplots(1, 1, figsize=(size * (cols + 1.5), size * (rows + 1.5)))
        defkwargs['ax'] = ax
    kwargs = defkwargs|kwargs
    sns.heatmap(matrix, **kwargs)
    
def plotNPMIMatrix(matrix, **kwargs):
    matrix, mask = (matrix[0], matrix[1]==0) if type(matrix) is tuple else (matrix, None)
    defkwargs = {
        'vmin': -0.2,
        'vmax': 0.2,
        'center': 0
    }
    kwargs = defkwargs|kwargs
    plotMatrix(matrix, mask=mask,
               **kwargs)


def generateResultsDF(results, vocab=base_vocab):
    resultslist = []
    for (train, test), preds in results['predictions'].items():
        partiallist = []
        for n in range(len(preds)):
            for pred in zip(preds[n][0], preds[n][1], preds[n][2]):
                partiallist.append({
                    'train': train,
                    'test': test,
                    'run': n,
                    'image': pred[0],
                    'y_true': pred[1].item(),
                    'y_pred': pred[2].item(),
                    'l_true': vocab[pred[1].item()],
                    'l_pred': vocab[pred[2].item()]
                })
        partialDF = pd.DataFrame(partiallist)
        ds = FERDataset2(test, load=False).df
        resultslist.append(partialDF.merge(ds, left_on='image', right_on='img_path'))
    resultsDF = pd.concat(resultslist)
    return resultsDF

def displayDF(df, style='byrow', cmap='YlGnBu'):
    if style == 'plain':
        display(df)
        return
    elif style == 'byrow':
        styler = df.round(2).style.format('{:.2f}').background_gradient(cmap=cmap, axis=1)
        display(styler)
    elif style == 'bycol':
        styler = df.round(2).style.format('{:.2f}').background_gradient(cmap=cmap, axis=0)
        display(styler)
    elif style == 'global':
        styler = df.round(2).style.format('{:.2f}').background_gradient(cmap=cmap, axis=None)
        display(styler)