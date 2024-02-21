

#%%
!git clone https://github.com/sizhky/zero-shot-learning/
!pip install -Uq torch_snippets
%cd zero-shot-learning/src

#%%
import gzip
import _pickle as cPickle
from torch_snippets import *
from sklearn.preprocessing import LabelEncoder, normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% define path to feature data and word2vec
WORD2VECPATH = "../data/class_vectors.npy"
DATAPATH = "../data/zeroshot_data.pkl"

#%% extract list of classes
with open("train_classes.txt", "r") as infile:
    train_classes = [str.strip(line) for line in infile]
    
# with load feature vector
with gzip.GzipFile(DATAPATH, "rb") as infile:
    data = cPickle.load(infile)

#%% define train data and zero shot data - classes that are present during training
training_data = [instance for instance in data if instance[0] in train_classes]
zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
np.random.shuffle(training_data) 

#%% 
train_size = 300 
train_data, valid_data = [], []
for class_label in train_classes:
    ctr = 0
    for instance in training_data:
        if instance[0] == class_label:
            if ctr < train_size:
                train_data.append(instance)
            else:
                valid_data.append(instance)
    
#%% shuffle training and validation data and fetch vectors corresponding to classes
np.random.shuffle(train_data)
np.random.shuffle(valid_data)
vectors = {i:j for i,j in np.load(WORD2VECPATH, allow_pickle=True)}

#%% fetch the image and word embedding features
train_data = [(feat,vectors[clss]) for clss,feat in train_data]
valid_data = [(feat,vectors[clss]) for clss, feat in valid_data]

#%% fetch training, validation and zero-shot classes
train_clss = [clss for clss, feat in train_data]
valid_clss = [clss for clss, feat in valid_data]
zero_shot_data = [clss for clss, feat in zero_shot_data]

#%%
x_train, y_train = zip(*train_data)
x_train, y_train = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
x_train = normalize(x_train, norm="l2")
x_valid, y_valid = zip(*valid_data)
x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
x_valid = normalize(x_valid, norm="l2")
y_zsl, x_zsl = zip(*zero_shot_data)
x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
x_zsl = normalize(x_zsl, norm="l2")

#%% define dataset and dataloader
from torch.utils.data import TensorDataset

trn_ds = TensorDataset(*[torch.Tensor(t).to(device) for t in [x_train, y_train]])
val_ds = TensorDataset(*[torch.Tensor(t).to(device) for t in [x_valid, y_valid]])

trn_dl = DataLoader(dataset=trn_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=32, shuffle=False)

#%%
def build_model():
    return nn.Sequential(
                        nn.Linear(in_features=4096, out_features=1024),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(num_features=1024),
                        nn.Dropout(p=0.8),
                        nn.Linear(in_features=1024, out_features=512),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(num_features=512),
                        nn.Dropout(p=0.8),
                        nn.Linear(in_features=512, out_features=256),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(num_features=256),
                        nn.Dropout(p=0.8),
                        nn.Linear(in_features=256, out_features=300),
                    )
    
    #%%  define train func
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, labels = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, labels = data
    _preds = model(ims)
    loss = criterion(_preds, labels)
    return loss.item()

#%% train the model
model = build_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
n_epochs = 60

log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss = train_batch(model, data, optimizer, criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss, end="\r")
        
    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss = validate_batch(model, data, criterion)
        log.record(ex+(bx+1)/N, val_loss=loss, end="\r")
        
    if not (ex+1)%10: log.report_avgs(ex+1)
    
log.plot_epochs(log=True)

#%% predict on images that contain zero-shot classes
pred_zsl = model(torch.Tensor(x_zsl).to(device)).cpu().detach().numpy()
class_vectors = sorted(np.load(WORD2VECPATH, allow_pickle=True), key=lambda x: x[0])
classnames, vectors = zip(*class_vectors)
classnames = list(classnames)
vectors = np.array(vectors)

#%% 
"""
Calculate the distance between each predicted vector and the vector
corresponding to the available classes and measure the number of zero-shot 
classes present in the top five predictions:
"""

dists = (pred_zsl[None] - vectors[:,None])
dists = (dists**2).sum(-1).T
best_classes = []

for item in dists:
    best_classes.append([classnames[j] for j in np.argsort(item)[:5]])
    
np.mean([i in j for i,j in zip(zero_shot_clss, best_classes)])
