from typing import (
    Tuple, 
    Union,
    List, 
    Optional,
    Any,
    cast,
)

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Conv2d,
    MaxPool2d,
    BatchNorm2d,
    ReLU,
    Sequential,
)
from torch.nn.functional import mse_loss

from torchvision.transforms import Normalize
from torchvision.models.vgg import (
    VGG,
    cfgs,
    VGG19_Weights
)

from torchvision.models._utils import (
    _ovewrite_named_param,
    handle_legacy_interface,
)
from torchvision.models._api import (
    WeightsEnum,
    register_model,
)


def _get_mean_and_std_per_channel(input: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    Return mean and std per channel of provided feature.

    Args:
        input (Tensor): Input feature to compute.
        eps (float, optional): Smooth term. Defaults to 1e-5.


    Raises:
        ValueError: Input does not have 4D dimention.


    Returns:
        Tuple[Tensor, Tensor]: mean and std
        
        
    Shape:
        Inputs:
            input: :math:`(N, C, H, W)`
        
        Outputs:
            mean, std: :math:`(N, C, 1, 1)`
    """
    
    if input.dim() != 4:
        raise ValueError(f'expected 4D input (got {input.dim()}D)')
    mean = torch.mean(input, dim=[-1, -2], keepdim=True) # (N, C, 1, 1)
    std = torch.std(input, dim=[-1, -2], keepdim=True) # (N, C, 1, 1)
    
    return mean, std


def gram_matrix(
    input: Tensor, 
    eps: float = 1e-5
):
    N, C, H, W = input.size()  # N = batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(N * C, H * W)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(N * C * H * W + eps)


class _VGG19Criterion(VGG):
    transforms = Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    ) 
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    
    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.eval()
        self.requires_grad_(False)
    
    
    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        content_reprs = []
        style_reprs = []
        
        z = self.transforms(x)
        # z = x
        
        i = 0
        for layer in self.features:
            z = layer(z)
            if isinstance(layer, Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, ReLU):
                name = f"relu_{i}"
            elif isinstance(layer, MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            if name in self.content_layers_default:
                content_reprs.append(z)
            if name in self.style_layers_default:
                style_reprs.append(z)
        
        return content_reprs, style_reprs        


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> Sequential:
    layers: List[Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU()]
            else:
                layers += [conv2d, ReLU()]
            in_channels = v
    return Sequential(*layers)


def _vgg_criterion(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = _VGG19Criterion(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


def _vgg19_criterion(*, weights: Optional[VGG19_Weights] = None, progress: bool = False, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    weights = VGG19_Weights.verify(weights)

    return _vgg_criterion("E", False, weights, progress, **kwargs)
        
        
 # return the pastiche and style-transfered latent
    
    
class StyleTransferLoss(Module):
    r"""
    
    Implement of Style Transfer loss, according to the paper "A Neural Algorithm of Artistic Style".
    For more details, https://arxiv.org/pdf/1508.06576
    
    """
    def __init__(
        self, 
        content_img: Tensor,
        style_img: Tensor,
        content_weight: float = 1,
        style_weight: float = 1000000,
        device=None,
        style_loss_method: str = "gram"
    ) -> None:
        r"""Initializer of loss object

        Args:
            content_img (Tensor): content image to calculate content loss.
            style_img (Tensor): style image to calculate style loss.
            content_weight (float, optional): Weight of content loss . Defaults to 1.
            style_weight (float, optional): Weight of style loss. Defaults to 1000000.
            style_loss_method (str): method to compute style loss. Can be either 'gram' or
                'stat'. Default to 'gram'.
        """
        super().__init__()
        assert(
            style_loss_method in ("gram", "stat")
        ), f"style_loss_method must be one of these: ('gram', 'stat'), got {style_loss_method}."
        self.feat_extractor: _VGG19Criterion = _vgg19_criterion(weights=VGG19_Weights.DEFAULT).to(device)
        
        content_reprs, _ = self.feat_extractor(content_img)
        for i, content_repr in enumerate(content_reprs):
            content_reprs[i] = content_repr.clone().detach()
        self.content_reprs = content_reprs
        
        _, style_reprs = self.feat_extractor(style_img)
        for i, style_repr in enumerate(style_reprs):
            style_reprs[i] = style_repr.clone().detach()
        self.style_reprs = style_reprs
        
        self.content_weight: float = content_weight
        self.style_weight: float = style_weight
        self.style_loss_method = style_loss_method
    
    
    def _content_loss_one_layer(
        self,
        pastiche_repr: Tensor,
        content_repr: Tensor
    ) -> Tensor:
        r"""
        Calculate content loss, based on pastiche's tensor of representation
        and content representation.
        
        
        Args:
            content_repr (Tensor): features of content image.
            pastiche_repr (Tensor): features of pastiche image.
            
        
        Returns:
            content_loss: Non-reduced loss tensor for a batch of inputs. 
            
        Shape:
            Inputs:
                content_repr: :math:`(N, C, H, W)`
                pastiche_repr: :math:`(N, C, H, W)`
            Outputs:
                content_loss: :math:`(N, )`
        """
        return mse_loss(pastiche_repr, content_repr) # Euclidean distance
        
    
    def _compute_content_loss(
        self, 
        pastiche_reprs: Tensor,
        content_reprs: Tensor
    ) -> Tensor:
        r"""
        Calculate content loss, based on pastiche's tensors of representation
        and content representations.
        
        
        Args:
            content_reprs (Tensor): features of content image.
            pastiche_reprs (Tensor): features of pastiche image.
            
        
        Returns:
            content_loss: Non-reduced loss tensor for a batch of inputs. 
            
        Shape:
            Inputs:
                content_repr: :math:`(N, C, H, W)`
                pastiche_repr: :math:`(N, C, H, W)`
            Outputs:
                content_loss: :math:`(N, )`
        """
        loss = 0.
        for pastiche_repr, content_repr in zip(pastiche_reprs, content_reprs):
            loss = loss + self._content_loss_one_layer(pastiche_repr, content_repr)
        return loss

    
    def _style_loss_one_layer(
        self,
        pastiche_repr: Tensor,
        style_repr: Tensor
    ) -> Tensor:
        r"""Compute style loss at one layer of the feature extractor.


        Args:
            style_repr (Tensor): features of style image.
            pastiche_repr (Tensor): features of pastiche image.


        Returns:
            style_loss: Non-reduced loss tensor for a batch of inputs at one
                layer of feature extractor. 
            
            
        Shape:
            Inputs:
                style_repr: :math:`(N, C, Hs, Ws)`
                pastiche_repr: :math:`(N, C, Hp, Wp)`
            Outputs:
                style_loss: :math:`(N, )`
        """
        if self.style_loss_method == "gram":
            G_pastiche = gram_matrix(pastiche_repr)
            G_style = gram_matrix(style_repr)
            return mse_loss(G_pastiche, G_style)
        else:
            mean_pastiche, std_pastiche = _get_mean_and_std_per_channel(pastiche_repr)
            mean_style, std_style = _get_mean_and_std_per_channel(style_repr)
            return mse_loss(mean_pastiche, mean_style) + mse_loss(std_pastiche, std_style)        
        
        
    def _compute_style_loss(
        self,
        pastiche_reprs: Tuple[Tensor],
        style_reprs: Tuple[Tensor]
    ) -> Tensor:
        r"""
        Compute style loss at multiple layers of the feature extractor.


        Args:
            style_reprs (Tuple[Tensor]): multiples features of style image.
            pastiche_reprs (Tuple[Tensor]): multiples features of pastiche image.


        Returns:
            style_loss: Non-reduced loss tensor for a batch of inputs at multiple
                layers of feature extractor.
        
        
        Shape:
            Ouputs:
                style_loss_sum: :math:`(N, )`        
        """
        loss = 0.
        for pastiche_repr, style_repr in zip(pastiche_reprs, style_reprs):
            loss = loss + self._style_loss_one_layer(pastiche_repr, style_repr)
        return loss
        
    
    def forward(
        self,
        pastiche: Tensor,
    ) -> Tensor:
        """Calculte Style Transfer loss. Return a reduced tensor.

        Args:
            pastiche (Tensor): Generated pastiche.

        Returns:
            loss: Reduced loss of one batch.
        """
        # get representations of pastiche and style image
        pastiche_reprs = self.feat_extractor(pastiche)

        # calculate content loss
        L_c = self._compute_content_loss(pastiche_reprs[0], self.content_reprs)
        
        # calculate style loss
        L_s = self._compute_style_loss(pastiche_reprs[1], self.style_reprs)
        
        # return content loss and style loss
        return self.content_weight*L_c, self.style_weight*L_s