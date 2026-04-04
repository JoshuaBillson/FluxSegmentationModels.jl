# FluxSegmentationModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/FluxSegmentationModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/FluxSegmentationModels.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/FluxSegmentationModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/FluxSegmentationModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/FluxSegmentationModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/FluxSegmentationModels.jl)

[FluxSegmentationModels](https://github.com/JoshuaBillson/FluxSegmentationModels.jl) is a pure Julia package implementing various semantic segmentation models in [Flux](https://fluxml.ai/Flux.jl/stable/).

## Available Models

| Model     | Source                                                                                                                                        | Implemented        |
| :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| U-Net     | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.1505.04597)                                  | :white_check_mark: |
| FPN       | [Feature Pyramid Networks for Object Detection](https://doi.org/10.48550/arXiv.1612.03144)                                                    | :white_check_mark: |
| SegFormer | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://doi.org/10.48550/arXiv.2105.15203)               | :white_check_mark: |
| SETR      | [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://doi.org/10.48550/arXiv.2012.15840)       | :white_check_mark: |

## Available Encoders

| Model    | Source                                                                                                                                        | Implemented        |
| :------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| ResNet   | [Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385)                                                     | :white_check_mark: |
| ConvNeXt | [A ConvNet for the 2020s](https://doi.org/10.48550/arXiv.2201.03545)                                                                          | :white_check_mark: |
| ViT      | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)                       | :white_check_mark: |
