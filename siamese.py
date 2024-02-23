

#%%
from torch_snippets import *

#%%
!wget https://www.dropbox.com/s/ua1rr8btkmpqjxh/face-detection.zip
#!unzip face-detection.zip

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

#%% define the dataset class

class SiameseNetworkDataset(Dataset):
    def __init__(self, folder, transform=None, should_invert=True):
        self.folder = folder
        self.items = Glob(f"{self.folder}/*/*")
        self.transform = transform
        
    def __getitem__(self, index):
        itemA = self.items[index]
        person = fname(parent(itemA))
        same_person = randint(2)
        if same_person:
            itemB = choose(Glob(f"{self.folder}/{person}/*", silent=True))
        else:
            while True:
                itemB = choose(self.items)
                if person != fname(parent(itemB)):
                    break
        imgA = read(itemA)
        imgB = read(itemB)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, np.array([1-same_person])
    
    def __len__(self):
        return len(self.items)


#%% transformation to prepare dataset and loaders
from torchvision import transforms
trn_tfms = transforms.Compose([transforms.ToPILImage(),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomAffine(degrees=5,
                                                       translate=(0.01, 0.2),
                                                       scale=(0.9, 1.1)
                                                       ),
                               transforms.Resize(size=(100,100)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5), std=(0.5))
                               ])
val_tfms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(100,100)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5), std=(0.5))
                               ])

trn_ds = SiameseNetworkDataset(folder=".data/faces/training/", transform=trn_tfms)
val_ds = SiameseNetworkDataset(folder="./data/faces/testing/", transform=val_tfms)

trn_dl = DataLoader(dataset=trn_ds, shuffle=True, batch_size=64)
val_dl = DataLoader(dataset=val_ds, shuffle=False, batch_size=64)

#%% define arch

def convBlock(ni, no):
    return nn.Sequential(nn.Dropout(p=0.2),
                         nn.Conv2d(in_channels=ni, out_channels=no, kernel_size=3, padding=1, padding_mode="reflect"),
                         nn.ReLU(inplace=True),
                         nn.BatchNorm2d(num_features=no),
                         )
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(convBlock(1,4),
                                      convBlock(4,8),
                                      convBlock(8,8),
                                      nn.Flatten(),
                                      nn.Linear(in_features=8*100*100, out_features=500),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(in_features=500, out_features=500),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(in_features=500, out_features=5)
                                      )
    
    def forward(self, input1, input2):
        output1 = self.features(input1)
        output2 = self.features(input2)
        return output1, output2
    









# %%
