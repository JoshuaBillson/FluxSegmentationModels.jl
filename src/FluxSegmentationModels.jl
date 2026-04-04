module FluxSegmentationModels

import Flux, Metalhead
using Match, ArgCheck, PartialFunctions
using Pipe: @pipe

Flux._show_leaflike(::Tuple{}) = true

include("utils.jl")

include("layers.jl")

include("encoders/interface.jl")
include("encoders/resnet.jl")
include("encoders/vit.jl")
include("encoders/convnext.jl")
export ResNet, ViT, ConvNeXt

include("decoders/fpn.jl")
include("decoders/unet.jl")
include("decoders/segformer.jl")
include("decoders/setr.jl")
#export FPNDecoder, UNetDecoder, SegFormerDecoder, SETRDecoder

include("models/unet.jl")
include("models/fpn.jl")
include("models/setr.jl")
include("models/segformer.jl")
export UNet, FPN, SETR, SegFormer

end