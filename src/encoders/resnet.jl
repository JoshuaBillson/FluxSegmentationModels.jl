"""
    ResNet(;depth=50, pretrain=false)

Configuration for constructing a ResNet encoder.

# Parameters
- `depth`: The depth of the ResNet architecture. One of `18`, `34`, `50`, `101`, or `152`.
- `pretrain`: If true, the ResNet encoder will be initialized with pretrained weights from ImageNet. Otherwise, the encoder will be randomly initialized.
"""
struct ResNet <: EncoderConfig
    depth::Int
    pretrain::Bool
end

function ResNet(;depth=50, pretrain=false)
    @argcheck depth in (18,34,50,101,152)
    return ResNet(depth, pretrain)
end

function encoder_block_dims(e::ResNet)
    @match e.depth begin
        18 || 34 => (64, 128, 256, 512)
        50 || 101 || 152 => (256, 512, 1024, 2048)
    end
end

function build_encoder(e::ResNet; inchannels=3)
    resnet = _build_resnet(e.depth, inchannels, e.pretrain)
    Flux.Chain(
        encoder = Flux.Chain(
            stem = resnet.layers[1][1], 
            body = resnet.layers[1][2:5]
        ), 
        head = resnet.layers[2]
    )
end

function _build_resnet(depth::Int, inchannels::Int, pretrain::Bool)
    return Metalhead.ResNet(depth; inchannels, pretrain)
end