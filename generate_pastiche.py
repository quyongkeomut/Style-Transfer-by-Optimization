SEED = 42

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch    
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = True
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

from lightning.pytorch import seed_everything
seed_everything(SEED, workers=True, verbose=False)

from utils.utils import _generate_pastiche, image_loader

from torchvision.transforms import ToPILImage


def main(
    experiment_name: str,
    content_path: str | Path,
    style_path: str | Path,
    num_steps: int,
    save_steps: int,
    content_weight,
    style_weight,
    style_loss_method
) -> None:
    r"""
    Perform creating pastiche, plot generated image and save it.
    """
    content_image = image_loader(content_path)
    style_image = image_loader(style_path)
    
    pastiche: torch.Tensor = _generate_pastiche(
        experiment_name=experiment_name,
        content_img=content_image,
        style_img=style_image,
        num_steps=num_steps,
        save_steps=save_steps,
        content_weight=content_weight,
        style_weight=style_weight,
        style_loss_method=style_loss_method
    )
    
    # save the last pastiche 
    pastiche_pil = ToPILImage()(pastiche.squeeze(0))
    save_path = f"results/{experiment_name}"
    os.makedirs(save_path, exist_ok=True)
    pastiche_pil.save(os.path.join(save_path, f"final_result.jpeg"))
    

if  __name__ == "__main__":                 
    parser = ArgumentParser(
        prog="Style transfer program",
        description="Provide a framework to generate pastiche from provided content image and style image"
    )
    
    # experiment id
    parser.add_argument("--exp_name", required=True, help="Name of experiment", type=str)   
    # content path
    parser.add_argument("--content_path", required=True, help="Path of content image", type=str)   
    # style path
    parser.add_argument("--style_path", required=True, help="Path of style image", type=str)
    # num steps
    parser.add_argument("--num_steps",  required=False, help="Max iterations for creating pastiche", default=300, type=int)
    # save steps
    parser.add_argument("--save_steps",  required=False, help="Saving length", default=100, type=int)
    # content weight
    parser.add_argument("--content_weight", required=False, help="Weight of content loss", default=1, type=float)
    # style weight
    parser.add_argument("--style_weight", required=False, help="Weight of style loss", default=1000000, type=float)
    # style loss calculation method 
    parser.add_argument("--style_loss_method", required=False, help="Method of style loss calculation", default="gram", type=str)
    
    args = parser.parse_args()
    
    main(
        experiment_name=args.exp_name,
        content_path=args.content_path,
        style_path=args.style_path,
        num_steps=args.num_steps,
        save_steps=args.save_steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        style_loss_method=args.style_loss_method
    )