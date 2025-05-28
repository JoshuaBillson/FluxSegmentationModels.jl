"""
    EncoderConfig

Super type of all encoder models. The type parameter `F` denotes the number of features
produced by each block of the encoder.

# Example Implementation
```julia
struct ResNet <: EncoderConfig
    depth::Int
    pretrain::Bool
end

function ResNet(;depth=50, pretrain=true)
    @argcheck depth in (18,34,50,101,152)
    return ResNet(depth, pretrain)
end

function encoder_features(e::ResNet)
    @match e.depth begin
        18 || 34 => [EncoderFeature(64,4), EncoderFeature(128,8), EncoderFeature(256,16), EncoderFeature(512,32)]
        50 || 101 || 152 => [EncoderFeature(256,4), EncoderFeature(512,8), EncoderFeature(1024,16), EncoderFeature(2048,32)]
    end
end

function build_encoder(e::ResNet; inchannels=3)
    resnet = _build_resnet(e.depth, inchannels, e.pretrain)
    Flux.Chain(
        Flux.Chain(resnet[1]..., resnet[2]...),
        resnet[3],
        resnet[4],
        resnet[5],
    )
end

function _build_resnet(depth::Int, inchannels::Int, pretrain::Bool)
    return Metalhead.ResNet(depth; inchannels, pretrain).layers[1]
end
```
"""
abstract type EncoderConfig end

"""
    build_encoder(encoder::EncoderConfig)

Constructs an encoder model based on the provided `EncoderConfig` configuration.

# Parameters
- `encoder`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built.

# Returns
A standard `Flux.Chain` layer containing each block of the encoder.
The returned encoder is ready to be integrated into more complex architectures like U-Net or used as a standalone feature extractor.

# Example
```julia
encoder = build_encoder(ResNet(depth=50, pretrain=true))
```
"""
function build_encoder end

struct EncoderFeature
    dim::Int
    scale::Int
end

"""
    encoder_features(encoder::EncoderConfig)

Returns the dimension and scale of the provided encoder.
"""
function encoder_features end

encoder_feature_dims(e::EncoderConfig) = [f.dim for f in encoder_features(e)]

encoder_feature_scales(e::EncoderConfig) = [f.scale for f in encoder_features(e)]