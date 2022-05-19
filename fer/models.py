from fastai.vision.all import *
import fastai.callback.schedule
import sklearn
import cloudpickle

from fer.paths import *

from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *

from fer.external_models.vgg_face_dag import *
from fer.external_models.vgg_m_face_bn_dag import *

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

class FERModel:
    def __init__(self, model_config, dls, dlsname, classweights=None):
        self.model_config = model_config
        self.pretrained = model_config['pretrained'] if 'pretrained' in model_config else True
        self.its = self.model_config['train_iterations']
        self.tta = self.model_config['tta']
        self.model_name = self.model_config['model_name']
        self.classweights = classweights
        self.mixedprecision = model_config['fp16'] if 'fp16' in model_config else False
        
        self.dls = dls
        self.dlsname = dlsname
        if self.model_name == 'vgg11':
            self.model = vgg11_bn
        elif self.model_name == 'vgg19':
            self.model = vgg19_bn
        elif self.model_name == "resnet18":
            self.model = resnet18
        elif self.model_name == "resnet34":
            self.model = resnet34
        elif self.model_name == 'resnet152-pretrained':
            self.model = resnet152
        elif self.model_name == "vgg_face_dag":
            self.model = partial(vgg_face_dag, weights_path=os.path.join(os.path.dirname(__file__), "external_models/vgg_face_models/vgg_face_dag.pth"))
        elif self.model_name == "vgg_m_face_bn_dag":
            self.model = partial(vgg_m_face_bn_dag, weights_path=os.path.join(os.path.dirname(__file__), "external_models/vgg_face_models/vgg_m_face_bn_dag.pth"))
        elif self.model_name == "densenet201":
            self.model = densenet201
        elif self.model_name == "efficientnet":
            self.model = partial(torch.hub.load, 'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4',)

        self.learner = cnn_learner(dls, self.model, 
                                pretrained=self.pretrained, 
                                path=LEARNERS_PATH, 
                                metrics=[accuracy],
                                loss_func=CrossEntropyLossFlat(weight=self.classweights))
        if self.mixedprecision:
            self.learner = self.learner.to_fp16()
        
    def train(self):
        self.learner.unfreeze()
        with self.learner.no_bar(): self.learner.fit_one_cycle(self.its)

    def plot_metrics(self):
        self.learner.recorder.plot_metrics()
        plt.show()
    
    def eval(self, dl, path):
        if self.tta:
            pred, y_true = self.learner.tta(dl=dl)
        else:
            pred, y_true = self.learner.get_preds(dl=dl, with_input=False, with_loss=False, with_decoded=False, act=None)
        y_pred = torch.argmax(pred, 1)
        if type(dl.items) == pd.DataFrame:
            paths = [str(x) for x in dl.items['img_path']]
        else:
            paths = [str(Path(x).relative_to(path)) for x in dl.items]
        res = (paths, y_true, y_pred, pred)
        return res
    
    def evalDF(self, dl, path):
        if self.tta:
            pred, y_true = self.learner.tta(dl=dl)
        else:
            pred, y_true = self.learner.get_preds(dl=dl, with_input=False, with_loss=False, with_decoded=False, act=None)
        y_pred = torch.argmax(pred, 1)
        if type(dl.items) == pd.DataFrame:
            paths = [str(x) for x in dl.items['img_path']]
        else:
            paths = [str(Path(x).relative_to(path)) for x in dl.items]
        res = (paths, y_true, y_pred, pred)
        df = pd.DataFrame(res, index=['image', 'y_true', 'y_pred', 'logits']).T
        return df
    
    def save(self, expname):
        cloudpickle.dump(self, open(f'{LEARNERS_PATH}/{expname}_{self.dlsname}.pkl', "wb"))
        
    def load(expname, dlsname):
        return cloudpickle.load(open(f'{LEARNERS_PATH}/{expname}_{dlsname}.pkl', "rb"))