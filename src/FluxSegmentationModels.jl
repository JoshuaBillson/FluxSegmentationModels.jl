module FluxSegmentationModels

import Flux, Metalhead
using Match, ArgCheck
using Pipe: @pipe

include("layers.jl")

include("encoders/interface.jl")
include("encoders/resnet.jl")
export ResNet

include("decoders/fpn.jl")
include("decoders/unet.jl")
include("decoders/deeplab.jl")
include("decoders/segformer.jl")
include("decoders/setr.jl")
export FPNDecoder, UNetDecoder, SegFormerDecoder

include("models/base.jl")
include("models/unet.jl")
export UNet

# Write your package code here.
function generate_features(encoder_dims, encoder_scales; imsize=256)
    return tuple([rand(Float32, imsize ÷ scale, imsize ÷ scale, dim, 1) for (dim, scale) in zip(encoder_dims, encoder_scales)]...)
end

end
