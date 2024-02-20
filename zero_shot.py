

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