"""
    SegFormer(encoder_config::EncoderConfig; embed_dim=768, dropout=0.0, nclasses=1, inchannels=3) where N

Construct a SegFormer style segmentation model.

# Parameters
- `encoder_config`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built and used in the U-Net.
- `embed_dim`: The embedding dimension to use for the decoder. Default is `256`.
- `dropout`: The dropout probability to use after the last layer of the decoder. Default is `0.0`.
- `nclasses`: The number of output classes for the segmentation task.
- `inchannels`: The number of channels in the input image.
"""
struct SegFormer{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand SegFormer

function SegFormer(encoder_config::EncoderConfig; inchannels=3, kw...)
    return SegFormer(
        extract_encoder(build_encoder(encoder_config; inchannels)), 
        encoder_block_dims(encoder_config); 
        kw...
    )
end

function SegFormer(encoder::Flux.Chain, encoder_dims::NTuple{N,Int}; embed_dim=256, dropout=0.0, nclasses=1) where N
    # Check Arguments
    @argcheck nclasses > 0

    # Build Decoder and Head
    decoder = SegFormerDecoder(encoder_dims; embed_dim, dropout)
    head = SegmentationHead(embed_dim, nclasses)
    return SegFormer(encoder, decoder, head)
end

function (m::SegFormer)(x::AbstractArray{<:Real,4})
    features = Flux.activations(m.encoder, x)
    decoder_out = m.decoder(features)
    return Flux.upsample_bilinear(decoder_out, size=feature_size(x)) |> m.head
end