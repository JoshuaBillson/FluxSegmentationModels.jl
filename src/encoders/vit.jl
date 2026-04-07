"""
    ViT(;config=:base, patchsize=(16,16), imsize=(256,256), dropout_prob=0.1, mlp_ratio=4.0, qkv_bias=true, pretrain=true)

Construct a ViT style encoder configuration.

# Parameters
- `config`: The ViT configuration to use. Can be `:tiny`, `:small`, `:base`, `:large`, or `:huge`.
- `patchsize`: The patch size to use for the encoder stem. Default is `(16,16)`.
- `imsize`: The image size to use for the input to the encoder. Default is `(224,224)`.
- `pretrain`: Whether to use ImageNet pre-trained weights. Default is `false`.
"""
struct ViT <: EncoderConfig
    config::Symbol
    pretrain::Bool
    patchsize::NTuple{2,Int}
    imsize::NTuple{2,Int}
end

function ViT(;config=:base, patchsize=(16,16), imsize=(224,224), pretrain=false)
    return ViT(config, pretrain, patchsize, imsize)
end

function encoder_block_dims(e::ViT)
    @match e.config begin
        :tiny => ntuple(_ -> 192, 12)
        :small => ntuple(_ -> 384, 12)
        :base => ntuple(_ -> 768, 12)
        :large => ntuple(_ -> 1024, 24)
        :huge => ntuple(_ -> 1280, 32)
    end
end

function build_encoder(e::ViT; inchannels=3, nclasses=1000)
    return vit(
        e.config;
        inchannels,
        nclasses,
        e.imsize, 
        e.patchsize, 
        e.pretrain
    )
end

function vit(config::Symbol; imsize=(224,224), inchannels=3, patchsize=(16,16), nclasses=1000, pretrain=true)
    # Check arguments
    @argcheck config in [:tiny, :small, :base, :large, :huge]
    @argcheck nclasses > 0
    @argcheck inchannels > 0
    @argcheck all(imsize .> 0)
    @argcheck all(patchsize .> 0)
    @argcheck all(imsize .% patchsize .== 0) "Image size must be divisible by patch size."
    if pretrain
        @argcheck inchannels == 3 "Pre-trained ViT models only support 3 input channels. Set `pretrain=false` to use a custom number of input channels."
        @argcheck imsize == (224, 224) "Pre-trained ViT models only support an image size of (224, 224). Set `pretrain=false` to use a custom image size."
    end

    # Construct Encoder
    model = Metalhead.ViT(config; imsize, patch_size=patchsize, inchannels, pretrain)
    encoder = model.layers[1]

    # Construct Head
    head = (nclasses == 1000) ? model.layers[2] : Metalhead.ViT(config; nclasses, pretrain=false).layers[2]

    # Assemble Encoder and Head into a single Chain
    return Flux.Chain(
        encoder = Flux.Chain(
            stem = encoder[1:4], 
            body = encoder[5]
        ),
        head = Flux.Chain(x -> x[:,1,:], head)
    )
end