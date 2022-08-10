import numpy as np
import torch
import glob
import imgaug.augmenters as iaa
import pytorch_lightning as pl
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.feature_extraction import create_feature_extractor


torch.manual_seed(0)


class DiceLoss(torch.nn.Module):
    """
    class to compute the Dice Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
                
        # Flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        counter = (pred * mask).sum()  # Numerator       
        denum = pred.sum() + mask.sum() + 1e-8  # Denominator. Add a small number to prevent NANS
        dice =  (2*counter)/denum
        return 1 - dice


class custom_dataset(torch.utils.data.Dataset):
    """
    1 -> HGG

    0 -> LGG
    """
    def __init__(self):
        self.HGGimages_list_address = []#glob.glob("./Preprocessed/HGG/*/data/77.npy")
        self.HGGlabels_list = []#glob.glob("./Preprocessed/HGG/*/masks/77.npy")
        self.LGGimages_list_address = glob.glob("./Preprocessed/LGG/*/data/77.npy")
        self.LGGlabels_list = glob.glob("./Preprocessed/LGG/*/masks/77.npy")
        self.complete_dataset_images = self.HGGimages_list_address + self.LGGimages_list_address
        self.complete_dataset_labels = self.HGGlabels_list + self.LGGlabels_list
    
    def __getitem__(self, index):

        with open(self.complete_dataset_images[index], "rb") as img:
            ip_img = torch.from_numpy(np.load(img)).float()
        
        with open(self.complete_dataset_labels[index], "rb") as img:
            op_img = torch.from_numpy(np.load(img)).float()
        
        return ip_img, op_img
    
    def __len__(self):
        return len(self.complete_dataset_images)


class classifn_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # starting img shape is (176, 176)
        self.L1 = torch.nn.Conv2d(1, 4, 3)
        self.relu1 = torch.nn.ReLU()
        self.L2 = torch.nn.MaxPool2d(2, 2)
        self.L3 = torch.nn.Flatten(0)
        # after flattening the img shape is (87*87*4)
        self.out = torch.nn.Linear(4*87*87, 2)
        self.soft = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.L1(x)
        x = self.relu1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.out(x)
        return self.soft(x)


class tumor_classifn(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.model = MaskRCNN(backbone=backbone)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.loss_func = DiceLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_ind):
        x, y = batch
        return self.loss_func(self.model(x), y)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.loss_func(self.model(x), y)
    
    def configure_optimizers(self):
        return self.opt


total_data = custom_dataset()
lenght_dataset = total_data.__len__()

train_data, test_data = torch.utils.data.random_split(total_data, [int(0.6*lenght_dataset), lenght_dataset-int(0.6*lenght_dataset)], generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)


# TODO: Add this later :-
# 
# seq = iaa.Sequential([
#     iaa.Affine(scale=(0.9, 1.1), rotate=(-30, 30))
# ])

trainer = pl.Trainer(max_epochs=8)
model_pretrained = torchvision.models.resnet18(pretrained=True)
backbn = create_feature_extractor(model_pretrained)
model = tumor_classifn(backbone=backbn)
trainer.fit(model=model, train_dataloaders=train_loader)
trainer.test(model=model, dataloaders=test_loader)