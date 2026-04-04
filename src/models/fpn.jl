"""
    FPN(encoder_config; inchannels=3, nclasses=1, pyramid_dim=256, segmentation_dim=128, upsample_method=:nearest, merge_policy=:add, dropout=0.0)
    FPN(encoder, encoder_dims; nclasses=1, pyramid_dim=256, segmentation_dim=128, upsample_method=:nearest, merge_policy=:add, dropout=0.0)

A Feature Pyramid Network style decoder. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_config`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built and used in the FPN.
- `inchannels`: The number of channels in the input image. Default is `3` for RGB images.
- `nclasses`: The number of output classes for the segmentation task. Default is `1`.
- `pyramid_dim`: The size of the feature pyramid dimension. Default is `256`.
- `segmentation_dim`: The size of the segmentation dimension. Default is `128`.
- `upsample_method`: The method to use for upsampling. Can be `:nearest` or `:bilinear`.
- `merge_policy`: The policy to use for merging features. Can be `:add` or `:concat`.
- `dropout`: The dropout probability to use after the last layer.
"""
struct FPN{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand FPN

function FPN(encoder_config::EncoderConfig; inchannels=3, kw...)
    return FPN(
        extract_encoder(build_encoder(encoder_config; inchannels)), 
        encoder_block_dims(encoder_config); 
        kw...
    )
end

function FPN(
    encoder::Flux.Chain, encoder_dims::NTuple{N,Int}; nclasses=1, 
    pyramid_dim=256, segmentation_dim=128, upsample_method=:nearest, 
    merge_policy=:add, dropout=0.0) where N

    # Check Arguments
    @argcheck nclasses > 0

    # Build Decoder and Head
    return FPN(
        encoder, 
        FPNDecoder(encoder_dims; pyramid_dim, segmentation_dim, upsample_method, merge_policy, dropout), 
        SegmentationHead(segmentation_dim, nclasses)
    )
end

function (m::FPN)(x::AbstractArray{<:Real,4})
    features = Flux.activations(m.encoder, x)
    decoder_out = m.decoder(features)
    return Flux.upsample_bilinear(decoder_out, size=feature_size(x)) |> m.head
end
