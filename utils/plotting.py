import torch
from torch import Tensor

from torchvision.transforms import ToPILImage

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_image(
    image: Tensor, 
    title: str | None
) -> None:
    img = image.cpu().clone() # clone the tensor to not do changes on it
    img = img.squeeze(0) # remove the fake batch dimension
    img = ToPILImage()(img)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.pause(0.001)
        
