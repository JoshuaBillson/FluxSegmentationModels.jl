module FluxSegmentationModels

import Flux, Metalhead
using Match, ArgCheck
using Pipe: @pipe

include("layers.jl")

include("decoders/fpn.jl")
include("decoders/unet.jl")
include("decoders/deeplab.jl")
include("decoders/segformer.jl")
include("decoders/setr.jl")

# Write your package code here.

end
