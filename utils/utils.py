import os
from pathlib import Path
from PIL import Image

import torch
from torch import Tensor

from torch.optim import LBFGS

from torchvision.transforms import (
    Resize,
    ToTensor,
    Compose,
    InterpolationMode,
    ToPILImage
)

from utils.loss import StyleTransferLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMGSIZE = (512, 512) if torch.cuda.is_available() else (384, 384)
LOADER = Compose([
    Resize(
        IMGSIZE, 
        InterpolationMode.NEAREST
    ),  # scale imported image  
    ToTensor(),
])


def image_loader(image_path: str | Path) -> Tensor:
    image = Image.open(image_path).convert("RGB")
    # fake batch dimension required to fit network's input dimensions
    image: Tensor = LOADER(image).unsqueeze(0)
    image = image.to(DEVICE, torch.float) 
    return image
   

def _generate_pastiche(
    experiment_name: str,
    content_img: Tensor,
    style_img: Tensor,
    num_steps: int = 300,
    save_steps: int = 100,
    content_weight: int = 1,
    style_weight: int = 1000000,
    style_loss_method: str = "gram"
) -> Tensor:
    r"""
    This function will generate a pastiche image based on content image 
    and style image.

    Args:
        experiment_idx (int): Index of experiment.
        content_img (Tensor): Content image. Scaled to [0, 1] range.
        style_img (Tensor): Style image. Scaled to [0, 1] range
        num_steps (int, optional): Number of step for LBFGS optimizer 
            to optimize. Defaults to 350.
        content_weight (int, optional): Weight of content loss. 
            Defaults to 1.
        style_weight (int, optional): Weight of style loss. 
            Defaults to 100.
        style_loss_method (str): method to compute style loss. Can be 
            either 'gram' or 'stat'. Default to 'gram'.
    Returns:
        Tensor: Pastiche image generated that share the same content with
            content image and have the "texture" like style image.
    """
    # initialize the pastiche
    pastiche = content_img.clone().detach()
    pastiche.requires_grad_(True)
    
    # initialize the optimizer and criterion
    optimizer = LBFGS([pastiche])
    criterion = StyleTransferLoss(
        content_img=content_img,
        style_img=style_img,
        content_weight=content_weight,
        style_weight=style_weight,
        device=DEVICE,
        style_loss_method=style_loss_method
    ).to(device=DEVICE)
    criterion.requires_grad_(False)
    
    
    # loop for creating pastiche
    run = 0
    save_step = 0
    while run <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                pastiche.clamp_(0, 1)

            optimizer.zero_grad(set_to_none=True)
            content_loss, style_loss = criterion(pastiche)
            loss = content_loss + style_loss
            loss.backward()
            
            nonlocal run, save_step
            run += 1
            save_step += 1
            if run % 50 == 0:
                print(f"run {run}:")
                print(f"Style Loss : {style_loss.item():4f} Content Loss: {content_loss.item():4f}")
                print()

            return loss
        
        optimizer.step(closure)
        
        # save pastiche if the loop reach the saving pivot
        if save_step % save_steps == 0:
            with torch.no_grad():
                pastiche.clamp_(0, 1)
            pastiche_pil = ToPILImage()(pastiche.squeeze(0))
            save_path = f"results/{experiment_name}"
            os.makedirs(save_path, exist_ok=True)
            pastiche_pil.save(os.path.join(save_path, f"{save_step//save_steps}.jpeg"))
    
    # last correction
    with torch.no_grad():
        pastiche.clamp_(0, 1)
        
    return pastiche
    
    
