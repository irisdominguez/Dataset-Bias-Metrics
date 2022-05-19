from fastai.vision.all import *
import matplotlib.pyplot as plt
import PIL
from functools import *
import os
import pathlib
import pandas as pd

import dill as pickle

from fer.paths import *
from fer.misc import *

SEED = 42


###################################
######## TRAIN / VAL SPLIT ########

def _path_idxs(items, root, name, ignore_paths):
    parent = Path(root).joinpath(Path(name))
    def _inner(items, name):
        return mask2idxs(Path(root, o).parent.parent == parent for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

def PathSplitter(root_path='.', train_path='train', valid_path='valid', ignore_paths=[]):
    "Split `items` from the grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _path_idxs(o, root_path, train_path, ignore_paths),_path_idxs(o, root_path, valid_path, ignore_paths)
    return _inner

def ParentSplitter(train_name='train', valid_name='valid'):
    def _inner(o):
        return _parent_idxs(o, train_name),_parent_idxs(o, valid_name)
    return 

def _identity_split(o, id_func, valid_pct=0.2, valid_uids=None, seed=SEED):
    uids = list(set(map(id_func, o)))
    if seed is not None: torch.manual_seed(seed)
    if valid_uids is None:
        rand_uids = L(list(torch.randperm(len(uids)).numpy()))
        cut = int(valid_pct * len(uids))
        valid_uids = list(map(uids.__getitem__, rand_uids[:cut]))
    valid_mask = np.array(list(map(lambda x: id_func(x) in valid_uids, o)))
    valid_ids = np.where(valid_mask)[0]
    train_ids = np.where(valid_mask == False)[0]
    return train_ids, valid_ids


def RandomTrainSubset(o, func, part=1.0):
    train_idxs, val_idxs = func(o)
    if 0 <= part <= 1.0:
        tr = int(len(train_idxs) * part)
    else:
        tr = part
    train_subset = np.random.choice(train_idxs, size=tr, replace=False)
    return train_subset, val_idxs


#############################
######## LABEL FILES ########

def _load_affectnet_label_file():
    return pickle.load(open(f'{DATA_PATH}/affectnet/exp_labels.pkl', 'rb'))


#################################
######## LABEL FUNCTIONS ########

def _parent_unified_labeller(o):
    raw = Path(o).parent.name.lower()
    replacements = {
        'anger': 'angry',
        'happiness': 'happy',
        'neutrality': 'neutral',
        'sadness': 'sad'
    }
    return replacements.get(raw, raw)
    
def _affectnet_labeller(o, labels):
    key = os.path.join(*o.parts[-3:])
    expressions = {
        0: 'neutral', 
        1: 'happy', 
        2: 'sad', 
        3: 'surprise', 
        4: 'fear', 
        5: 'disgust', 
        6: 'angry', 
        7: 'contempt'
    }
    return expressions[labels[key]]


#####################
######## MISC ########

def CategoryBlock(vocab=None, sort=True, add_na=False):
    "`TransformBlock` for single-label categorical targets"
    return TransformBlock(type_tfms=Categorize(vocab=vocab, sort=sort, add_na=add_na))


#################################
######## DATASET CONFIGS ########

base_vocab = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

source_dataset_configs = {
    'affectnet': {
        'image_path': f'{DATA_PATH}/affectnet',
        'label_file': _load_affectnet_label_file,
        'labeller': _affectnet_labeller,
        'splitter': GrandparentSplitter(train_name='train_set', valid_name='val_set')},
    
    'ferplus': {
        'image_path': f'{DATA_PATH}/fer2013/images_from_ferplus',
        'splitter': GrandparentSplitter(train_name='train', valid_name='valid')},
    
}


class SourceDataset:
    def getImages(self):
        path = self.image_path
        filefilters = self.filefilters
        images = get_files(path, extensions=image_extensions, recurse=True, folders=None)
        for f in filefilters:
            images = list(filter(f, images))
        return images
    
    def __init__(self, name, bs=128, vocab=base_vocab, grayscale=False, target=None):
        self.bs = bs
        self.name = name
        self.vocab = vocab
        self.target = target
        
        self.basename = name
            
        config = source_dataset_configs[self.basename]
            
        if target is not None:
            if vocab is None:
                self.vocab = config['targets'][target]['vocab']
        
        self.image_path = config['image_path']
        self.labels = config['label_file']() if 'label_file' in config else None
        
        self.identity_function = config['identity_function'] if 'identity_function' in config else None
        self.safe_identity_function = md5 if self.identity_function is None else self.identity_function
        def safe_identity_function_wrapper(x, f):
            return f(self.filepath(x))
        self.safe_identity_function = partial(safe_identity_function_wrapper, f=self.safe_identity_function)
        
        self.sequence_function = config['sequence_function'] if 'sequence_function' in config else None
        self.safe_sequence_function = md5 if self.sequence_function is None else self.sequence_function
        
        if self.identity_function is not None:
            if 'valid_uids' in config:
                self.splitter = partial(_identity_split, id_func=self.identity_function, valid_uids=config['valid_uids'])
            else:
                self.splitter = partial(_identity_split, id_func=self.identity_function)
        else:
            self.splitter = config['splitter'] if 'splitter' in config else None
            
        if 'labeller' in config:
            self.labeller = config['labeller']
            if self.labels is not None:
                self.labeller = partial(self.labeller, labels=self.labels)
            if self.target is not None:
                self.labeller = partial(self.labeller, target=self.target)
        else:
            self.labeller = _parent_unified_labeller

        self.filefilters = []
        self.filefilters.append(lambda o: self.labeller(o) in self.vocab)
        
    def filepath(self, x, absolute=False):
        abspath = Path(self.image_path, x)
        if absolute:
            return str(abspath)
        else:
            relpath = abspath.relative_to(self.image_path)
            return str(relpath)
        
    def getPandas(self):
        images = self.getImages()
        images = list(map(lambda x: x.relative_to(self.image_path), images))
        
        df = pd.DataFrame(np.array(images), columns=['img_path'])
        df['subject'] = df['img_path'].map(lambda x: self.safe_identity_function(Path(x)))
        df['sequence'] = df['img_path'].map(lambda x: self.safe_sequence_function(Path(x)))
        
        df['label'] = df['img_path'].map(lambda x: self.labeller(Path(x)))
        
        paths = list(map(Path, df['img_path']))
        
        idx_train, idx_val = self.splitter(paths)
        df['partition'] = ''
        df.loc[idx_train, 'partition'] = 'train'
        df.loc[idx_val, 'partition'] = 'val'
        
        df['id'] = df['label'] + '_' + df['img_path'].map(md5)
        
        df = df[['id', 'img_path', 'partition', 'label', 'subject', 'sequence']]
        
        return df

class ConvertBW(Transform):
    def encodes(self, img: PILImage):
        return img.convert('L').convert('RGB')

class FERDataset2():
    def getItemTransforms(self):
        tfms = [Resize(224, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)]
        if self.grayscale:
            tfms.append(ConvertBW)
        return tfms
    
    def getBatchTransforms(self):
        return aug_transforms(size=112)
    
    def __init__(self, name, bs=128, vocab=base_vocab, grayscale=False, target=None, load=True, df=None, verbose=False):
        self.bs = bs
        self.name = name
        self.vocab = vocab
        self.grayscale = grayscale
        
        if df is None:
            path = f'{PROCESSED_PATH}/data/final/{name}.csv'
            if not os.path.exists(path):
                path = f'{PROCESSED_PATH}/data/composed_datasets/{name}.csv'
            self.df = pd.read_csv(path, dtype={'partition': 'str'})   
        else:
            self.df = df
            
        self.image_path = f'{DATA_PATH}/cropped'
        
        self.df['val'] = self.df['partition'] == 'val'
        self.df = self.df.set_index('img_path', drop=False)
        self.df.index.names = ['index']
        
        def safe_identity_function_wrapper(x, f):
            return f(self.filepath(x))
        self.safe_identity_function = partial(safe_identity_function_wrapper, f=str)
        
        if vocab is not None:
            self.df = self.df[self.df['label'].isin(vocab)]
        total = len(self.df)
        na = sum(self.df['cropped_img'].isna())
        if verbose: print(f'[{name}] Dropping {na} out of {total} rows with no cropped image ({100*na/total:.4f}%)')
        self.df = self.df[self.df['cropped_img'].notna()]
        
        self.dls = None
        if load:
            self.loadDLS()
            
    def labeller(self, o):
        return self.df.loc[o, 'label']
        
    def loadDLS(self, part=None):
        self.dls = ImageDataLoaders.from_df(self.df, 
                                             path = self.image_path,
                                             fn_col = 'cropped_img',
                                             label_col = 'label',
                                             valid_col = 'val',
                                             y_block = partial(CategoryBlock, vocab=self.vocab),
                                             item_tfms=self.getItemTransforms(),
                                             batch_tfms=self.getBatchTransforms(),
                                             bs=self.bs, 
                                             num_workers=8, 
                                             drop_last=False)
        
        return self.dls
    
    def getAggregatedCSV(self, 
                         aggregated='none',
                         conf_thr=0, 
                         cached=True):
        confs = {
            'race': 'race_conf_fair', 
            'race4': 'race_conf_fair_4', 
            'gender': 'gender_conf_fair', 
            'age': 'age_conf_fair'
        }
        
        df = self.df.copy()
        
        if ((aggregated == 'subject' or aggregated == 'subject-collapse') and
            not df['id'].equals(df['label'] + '_' + df['subject'].astype(str))):
            print(f'[{self.name}] Data aggregated by identity')
            
            df_agg = df.groupby(by='subject').agg(lambda x: x.value_counts().index[0])
            
            for ids in df_agg.index:
                filtered = df[df['subject'] == ids]
                for targetcol, confcol in confs.items():
                    value = df_agg.loc[ids][targetcol]
                    filtered2 = filtered[filtered[targetcol] == value]
                    conf = np.mean(filtered[confcol])
                    df_agg.loc[ids, confcol] = conf
            
            replace_cols = [val for pair in zip(list(confs.keys()), list(confs.values())) for val in pair]
            df_agg = df_agg[replace_cols]
            
            df = df.drop(replace_cols, axis=1)
            df = df.merge(df_agg, left_on='subject', right_index=True)
            
            df = df.drop([
                'age_scores_fair',
                'gender_scores_fair',
                'race_scores_fair_4',
                'race_scores_fair'], axis=1)
            
            if aggregated == 'subject-collapse':
                return df_agg

        if ((aggregated == 'sequence' or aggregated == 'sequence-collapse') and
            not df['id'].equals(df['label'] + '_' + df['subject'].astype(str))):
            print(f'[{self.name}] Data aggregated by sequence')
            df_agg = df.groupby(by=['subject', 'sequence']).agg(lambda x: x.value_counts().index[0])

            for subj, seq in df_agg.index:
                filtered = df[(df['subject'] == subj) & (df['sequence'] == seq)]
                for targetcol, confcol in confs.items():
                    value = df_agg.loc[(subj, seq)][targetcol]
                    filtered2 = filtered[filtered[targetcol] == value]
                    conf = np.mean(filtered[confcol])
                    df_agg.loc[(subj, seq), confcol] = conf
            
            replace_cols = [val for pair in zip(list(confs.keys()), list(confs.values())) for val in pair]
            
            df = df.drop(replace_cols, axis=1)
            df = df.merge(df_agg[replace_cols], left_on=['subject', 'sequence'], right_index=True)
            
            df = df.drop([
                'age_scores_fair',
                'gender_scores_fair',
                'race_scores_fair_4',
                'race_scores_fair'], axis=1)
            
            if aggregated == 'sequence-collapse':
                df_agg = df_agg[replace_cols + ['label']]
                return df_agg

        return df
        
    def show(self, saveSample=False):
        self.dls[0].show_batch(max_n=25)
        plt.subplots_adjust(top=0.95)
        plt.suptitle(f'{self.name}, train')
        
        if saveSample:
            plt.savefig(f'{SAMPLES_PATH}/{self.name}_train.png')
        plt.show()
            
        print('---------------------------------------------')
        
        self.dls[1].shuffle = True
        self.dls[1].show_batch(max_n=25)
        plt.subplots_adjust(top=0.95)
        plt.suptitle(f'{self.name}, test')
        if saveSample:
            plt.savefig(f'{SAMPLES_PATH}/{self.name}_test.png')
        plt.show()
        
    def getClassDistribution(self):
        df = self.df.copy()
        df = df[df['partition'] == 'train']
        df = df['label'].value_counts()
        for label in self.vocab:
            if label not in df:
                # Pad with 1 to allow calculation of class_weights
                df = df.append(pd.Series({label: 1}))
        df = df.loc[self.vocab]
        return df
    
    def size(self):
        train = len(self.df[self.df['partition']=='train'])
        val = len(self.df[self.df['partition']=='val'])
        return train, val
    
    def stratifiedSubsample(self, proportion, load=True):
        print(f'[{self.name}] Generating stratified subsample dataset [{proportion}]')
        self.df = pd.concat(
            [self.df[self.df['partition']=='train'].groupby('race', group_keys=False).apply(lambda x: x.sample(frac=proportion)),
             self.df[self.df['partition']=='val']]
        )
        if load:
            self.loadDLS()
        print(f'[{self.name}] Stratified subsample dataset [{proportion}] generated ')
        return self
    
    def filterToDemography(self, column, values, confcolumn=None, confthr=None):
        self.df = self.df[self.df[column].isin(values)]
        if confcolumn:
            self.df = self.df[self.df[confcolumn] > confthr]
