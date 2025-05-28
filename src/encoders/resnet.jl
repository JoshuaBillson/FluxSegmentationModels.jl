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