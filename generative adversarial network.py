import torch
torch.manual_seed(42)
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from torchsummary import summary
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

device="cuda"# image=image.to(device) i.e to transfer my image to gpu device
batch_size=128 # going to be used in trainloader and training loop
noise_dim=64 # to create generator model
# optimiser parameters
lr=0.002 #learning rate
beta_1=0.5 # beta 1 and 2 are going to be passed to atom optimizer
beta_2=0.99
# traingig variables
epochs=20 # basic number of times we want to run our loop



train_augs=T.Compose([T.RandomRotation((-20,+20)),T.ToTensor()])
            # by doing this our image will change to torch tensor with channel by heoght by width

trainset=datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)

image, label= trainset[0]
plt.imshow(image.squeeze(),cmap='gray')
print('The total number of images are :',len(trainset) )


trainloader=DataLoader(trainset, batch_size=batch_size,shuffle=True)
print('Total number of batches in this train loader:', len(trainloader))

dataiter=iter(trainloader)
#images, _ = dataiter.next()
#print(images.shape)

for images, _ in trainloader:
          print(images.shape)   

# 'show_tensor_images' : function is used to plot some of images from the batch

def show_tensor_images(tensor_img, num_images = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()


show_tensor_images(images, num_images=24)



def get_disc_block(in_channels, out_channels, kernel_size, stride):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2)
  )
      

class Discriminator(nn.Module):

  def __init__(self): # two dash use garne
    super(Discriminator, self).__init__()

    self.block_1=get_disc_block(1, 16, (3,3), 2)
    self.block_2=get_disc_block(16, 32, (5,5), 2)
    self.block_3=get_disc_block(32, 64, (5,5), 2)

    self.flatten=nn.Flatten()
    self.linear=nn.Linear(in_features=64, out_features=1)


  def forward( self, images):
    x1=self.block_1(images)
    x2=self.block_2(x1)
    x3=self.block_3(x2)

    x4= self.flatten(x3)
    x5=self.linear(x4)

    return x5
                    