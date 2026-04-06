"""
    ResNet(;depth=50, pretrain=false)

Configuration for constructing a ResNet encoder.

# Parameters
- `depth`: The depth of the ResNet architecture. One of `18`, `34`, `50`, `101`, or `152`.
- `pretrain`: If true, the ResNet encoder will be initialized with pretrained weights from ImageNet.
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

function build_encoder(e::ResNet; inchannels=3, nclasses=1000)
    # Construct Encoder
    model = Metalhead.ResNet(e.depth; inchannels, e.pretrain)
    encoder_stem = model.layers[1][1]
    encoder_body = model.layers[1][2:5]

    # Construct Head
    head = (nclasses == 1000) ? Metalhead.ResNet(e.depth; nclasses, pretrain=false).layers[2] : model.layers[2]

    # Assemble Encoder and Head into a single Chain
    Flux.Chain(
        encoder = Flux.Chain(
            stem = encoder_stem, 
            body = encoder_body
        ), 
        head = head
    )
end