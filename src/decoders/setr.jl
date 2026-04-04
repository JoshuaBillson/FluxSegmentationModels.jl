"""

    SETRDecoder(encoder_dim::Int; n_features=4, batch_norm=true, act=Flux.relu)

A decoder for the SETR model. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_dim`: The feature dimension of the output of the encoder.
- `n_features`: The number of features provided from the encoder. Default is `4`.
- `batch_norm`: If true, a batch norm operation will be applied after each convolution in the decoder. Default is `true`.
- `act`: The activation function to use after each convolution. Default is `Flux.relu`.

# Input
A tuple of WHCN or CLN arrays corresponding to the output of each block of a matching encoder ordered from first to last.
"""
struct SETRDecoder{P,S,U}
    pyramid::P
    fuse::S
    upsample::U
end

Flux.@layer :expand SETRDecoder

function SETRDecoder(encoder_dim::Int; n_features=4, batch_norm=true, act=Flux.relu)
    @argcheck encoder_dim > 0

    # Construct Feature Pyramid
    norm = batch_norm ? :BN : nothing
    feature_pyramid = Flux.Parallel(
        accumulate_features $ +, 
        [Conv((1,1), encoder_dim, encoder_dim÷2; act, norm) for _ in 1:n_features]...,
    )

    # Construct Segmentation Layers
    seg_blocks = Flux.Parallel(
        concatenate_features,
        [Flux.Chain(
            Conv((3,3), encoder_dim÷2, encoder_dim÷2; act, norm), 
            Conv((3,3), encoder_dim÷2, encoder_dim÷4; act, norm)
        ) for _ in 1:n_features]...,
    )

    # Construct Upsampling Layer
    upsample = Flux.Chain(
        Base.Fix2(Flux.upsample_bilinear, (4,4)),
        Conv((3,3), (encoder_dim ÷ 4) * n_features, encoder_dim; act, norm),
    )

    # Assemble Decoder
    return SETRDecoder(feature_pyramid, seg_blocks, upsample)
end

function (m::SETRDecoder)(features::NTuple{N,AbstractArray{<:Real,3}}, imsize::NTuple) where N
    return m(map(x -> seq2img(x, imsize), features))
end

function (m::SETRDecoder)(features::NTuple{N,AbstractArray{<:Real,4}}) where N
    return reverse(features) |> m.pyramid |> m.fuse |> m.upsample
end