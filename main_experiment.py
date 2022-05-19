import os
USE_GPUS = '0'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = USE_GPUS
N_GPUS = len(USE_GPUS.split(','))

from fastai.vision.all import *
import sklearn
from fer import *

import sys
 
EXP = 'exp_affectnet'
base_datasets = ['affectnet']
vocab=['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EXP = 'exp_ferplus'
base_datasets = ['ferplus']
vocab=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

f = open(f'{LOGS_PATH}/{EXP}.log', 'a')
sys.stdout = f
sys.stderr = f
print = partial(print, flush=True)
    
print(f'# Log for {EXP}')

bs = 256
grayscale = False

SAVE_RESULTS = f'{RESULTS_PATH}/{EXP}.pkl'
SAVE_RESULTS_DF = f'{RESULTS_PATH}/{EXP}.csv'

datasets = []
for dataset in base_datasets:
    for p in [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.60, 1.00]:
        datasets.append((f'{dataset}_sub_{p:.2f}', dataset))
    datasets.append((f'compdataset_{dataset}_race_balanced', dataset))
    datasets.append((f'compdataset_{dataset}_gender_balanced', dataset))
    datasets.append((f'compdataset_{dataset}_gender_biased_Male-only', dataset))
    datasets.append((f'compdataset_{dataset}_gender_biased_Female-only', dataset))

reps = 10

model_configs = {'vgg11': {
    'model_name': 'vgg11',
    'pretrained': False,
    'train_iterations': 100,
    'tta': False,
    'fp16': False
}}


for name, _ in datasets:
    for r in range(reps):
        for config in model_configs:
            if os.path.exists(f'{LEARNERS_PATH}/{EXP}_{r}_{config}_{name}.pkl'):
                print(f'> Skip [{name}][{r+1}/{reps}][{config}] (already trained)')
                continue
            print(f'> Train on [{name}][{r+1}/{reps}][{config}]')
            if '_sub_' in name:
                base_name = name.split('_sub_')[0]
                portion = float(name.split('_sub_')[1])
                ds = FERDataset2(base_name, vocab=vocab, grayscale=grayscale, load=False)
                ds.stratifiedSubsample(portion, load=True)
                dls = ds.dls
            else:
                ds = FERDataset2(name, vocab=vocab, grayscale=grayscale, load=True)
                dls = ds.dls

            classdistribution = ds.getClassDistribution()

            mock_y = np.hstack([np.ones(x)*i for i, x in enumerate(classdistribution)])
            classweights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(len(classdistribution)), y=mock_y)
            classweights = torch.FloatTensor(classweights).cuda()

            model = FERModel(model_configs[config], dls, name, classweights = classweights)
            model.train()
            model.save(f'{EXP}_{r}_{config}')
            print(f'< Trained on [{name}][{r+1}/{reps}][{config}]')

results = {
    'exp': EXP,
    'vocab': None,
    'reps': reps,
    'datasets': datasets,
    'model_configs': model_configs,
    'predictions': {}
}

resdfs = []

for train_name, test_name in datasets:
    if test_name != last_test_name:
        ds = FERDataset2(test_name)
        dls = ds.dls
        if results['vocab'] is None:
            results['vocab'] = dls.vocab
        else:
            # Should be the same for all datasets
            assert results['vocab'] == dls.vocab
        dls_valid = dls[1]
        last_test_name = test_name

    results['predictions'][(train_name, test_name)] = []
    for r in range(reps):
        for config in model_configs:
            print(f'> Eval learner trained on [{train_name}][{r+1}/{reps}][{config}] on test partition of [{test_name}]')
            model = FERModel.load(f'{EXP}_{r}_{config}', train_name)
            res = model.eval(dls_valid, dls.path)
            results['predictions'][(train_name, test_name)].append(res)

            print(f'> [{train_name} => {test_name}][{r+1}/{reps}][{config}] Evaluated')

if SAVE_RESULTS is not None:
    pickle.dump(results, open(SAVE_RESULTS, 'wb'))
    
resdfs = pd.concat(resdfs)
resdfs.to_csv(SAVE_RESULTS_DF, index=None)

print(f'Execution completed {EXP}')
