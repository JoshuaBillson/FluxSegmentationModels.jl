"""
    ViT(;config=:base, patchsize=(16,16), imsize=(256,256), dropout_prob=0.1, mlp_ratio=4.0, qkv_bias=true)

Construct a ViT style encoder configuration.

# Parameters
- `config`: The ViT configuration to use. Can be `:tiny`, `:small`, `:base`, `:large`, or `:huge`.
- `patchsize`: The patch size to use for the encoder stem. Default is `(16,16)`.
- `imsize`: The image size to use for the input to the encoder. Default is `(256,256)`.
- `dropout_prob`: The dropout probability to use for the encoder. Default is `0.1`.
- `mlp_ratio`: The MLP ratio to use for the encoder. Default is `4.0`.
- `qkv_bias`: Whether to use bias for the query, key, and value projections in the encoder. Default is `true`.
"""
struct ViT <: EncoderConfig
    config::Symbol
    patchsize::NTuple{2,Int}
    imsize::NTuple{2,Int}
    dropout_prob::Float64
    mlp_ratio::Float64
    qkv_bias::Bool
end

function ViT(;config=:base, patchsize=(16,16), imsize=(256,256), dropout_prob=0.1, mlp_ratio=4.0, qkv_bias=true)
    @assert config in [:tiny, :small, :base, :large, :huge]
    return ViT(config, patchsize, imsize, dropout_prob, mlp_ratio, qkv_bias)
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

function build_encoder(e::ViT; inchannels=3)
    return vit(
        e.config;
        inchannels,
        e.imsize, 
        e.patchsize, 
        e.mlp_ratio, 
        e.qkv_bias, 
        e.dropout_prob, 
        emb_dropout_prob=e.dropout_prob
    )
end

function vit(config::Symbol; kw...)
    @match config begin 
        :tiny => vit(192, 3, 12; kw...)
        :small => vit(384, 6, 12; kw...)
        :base => vit(768, 12, 12; kw...)
        :large => vit(1024, 16, 24; kw...)
        :huge => vit(1280, 16, 32; kw...)
    end
end

function vit(embedplanes::Integer, nheads::Integer, depth::Integer; imsize=(256, 256), 
    inchannels=3, patchsize=(16, 16), mlp_ratio=4.0, dropout_prob=0.1, emb_dropout_prob=0.1, 
    pool=:class, nclasses::Integer = 1000, qkv_bias = true)

    model = Metalhead.vit(
        imsize; inchannels, patch_size=patchsize, embedplanes, 
        depth, nheads, mlp_ratio, dropout_prob, 
        emb_dropout_prob, pool, nclasses, qkv_bias
    )

    encoder = model.layers[1]
    return Flux.Chain(
        encoder = Flux.Chain(
            stem = encoder[1:4], 
            body = encoder[5]
        ),
        head = model.layers[2]
    )
end