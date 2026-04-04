```@meta
CurrentModule = FluxSegmentationModels
```

# FluxSegmentationModels

Documentation for [FluxSegmentationModels](https://github.com/JoshuaBillson/FluxSegmentationModels.jl).


# Available Encoders

| Model    | Source                                                                                                                                        | Implemented        |
| :------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| ResNet   | [Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385)                                                     | :white_check_mark: |
| ConvNeXt | [A ConvNet for the 2020s](https://doi.org/10.48550/arXiv.2201.03545)                                                                          | :white_check_mark: |
| ViT      | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)                       | :white_check_mark: |

```@docs
ResNet
ConvNeXt
ViT
```

# Available Segmentation Models

| Model    | Source                                                                                                                                        | Implemented        |
| :------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| U-Net    | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.1505.04597)                                  | :white_check_mark: |
| FPN      | [Feature Pyramid Networks for Object Detection](https://doi.org/10.48550/arXiv.1612.03144)                                                    | :white_check_mark: |
| SETR     | [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://doi.org/10.48550/arXiv.2012.15840)       | :white_check_mark: |

```@docs
UNet
FPN
SETR
```