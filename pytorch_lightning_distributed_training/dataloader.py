import os 
import random 
from typing import Optional 


from pytorch_lightning import LightningDataModule 

import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader, Dataset 

import torch 

class synthetic_imagenet_test(Dataset): 
    def __len__(self): 
        return int(1e6)
    
    def __getitem__(self, item): 
        image= torch.rand(3, 224, 224)
        label= random.randint(0, 999)
        return image, label 

class ImageNetDataModule(LightningDataModule): 
    def __init__(self, data_path: Optional[str]="/data", 
                batch_size: int =4, 
                workers: int= 2, 
                **kwargs, 
                ): 
        super().__init__()
        self.data_path= data_path
        self.batch_size= batch_size 
        self.workers= workers 

        if self.data_path is None: 
            print(f"Using synthetic imagenet for testing -- Argument --data_path is not providing")
    
    def train_dataloader(self): 
        if self.data_path is None: 
            train_dataset= synthetic_imagenet_test() 
        else: 
            train_dir= os.path.join(self.data_path, "train")
            normalize= transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            train_dataset= ImageFolder(train_dir, transforms.Compose(

                [transforms.RandomResizedCrop(224), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                normalize, 
                ]
            )) 

            train_dataloader=DataLoader(dataset= train_dataset, 
                batch_size= self.batch_size, 
                shuffle= True, 
                num_workers= self.workers, 
            )
            return train_dataloader 

    def val_dataloader(self): 
        if self.data_path is None:
            val_dataset= synthetic_imagenet_test() 
        else: 
            val_dir= os.path.join(self.data_path, "val")
            normalize= transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            val_dataset= ImageFolder(val_dir, transforms.Compose([

            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            normalize, ]
            )
            )
        val_loader= DataLoader(val_dataset, batch_size= self.batch_size, 
                    shuffle= False , 
                    num_workers= self.workers, 
        )
        return val_loader 
    
    def test_dataloader(self): 
        return self.val_dataloader() 

    
class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

            
