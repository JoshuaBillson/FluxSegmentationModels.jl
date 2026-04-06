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

function build_encoder(e::ResNet; inchannels=3, nclasses=1000)
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
    build_encoder(encoder::EncoderConfig; inchannels=3, nclasses=1000)

Constructs an encoder model based on the provided `EncoderConfig` configuration.

# Parameters
- `encoder`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built.
- `inchannels`: The number of channels in the input image. Default is `3` for RGB images.

# Returns
A standard `Flux.Chain` layer containing each block of the encoder.
The returned encoder is ready to be integrated into more complex architectures like U-Net or used as a standalone feature extractor.

# Example
```julia
encoder = build_encoder(ResNet(depth=50, pretrain=true))
```
"""
function build_encoder end

"""
    encoder_block_dims(encoder::EncoderConfig)

Returns the dimensions of the provided encoder blocks.
"""
function encoder_block_dims end

"""
    extract_encoder(m::Flux.Chain)

Extracts the encoder blocks from a `Flux.Chain` layer containing an encoder and returns a new `Flux.Chain` layer containing only the encoder blocks.
"""
function extract_encoder(m::Flux.Chain)
    encoder = m.layers.encoder
    encoder_stem = encoder.layers.stem
    encoder_body = encoder.layers.body
    Flux.Chain(
        Flux.Chain(encoder_stem..., encoder_body[1]...),   # First block includes the stem
        encoder_body[2:end]...                             # Remaining blocks
    )
end