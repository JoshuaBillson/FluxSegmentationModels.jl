struct UNet{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand UNet

"""
    UNet(encoder_config; decoder_dims=(64,128,256,512,1024), batch_norm=true, upsample_method=:nearest, nclasses=1, inchannels=3)
    UNet(encoder, encoder_dims; decoder_dims=(64,128,256,512,1024), batch_norm=true, upsample_method=:nearest, nclasses=1)

Construct a U-Net style segmentation model.

# Parameters
- `encoder_config`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built and used in the U-Net.
- `encoder`: A `Flux.Chain` layer containing the blocks of the encoder to be used in the U-Net.
- `encoder_dims`: A tuple containing the feature dimension of each encoder block output ordered from first to last.
- `decoder_dims`: The feature dimension of each decoder block ordered from top to bottom.
- `batch_norm`: If true, a batch norm operation will be applied after each convolution in the decoder.
- `inchannels`: The number of channels in the input image.
- `nclasses`: The number of output classes for the segmentation task.
- `upsample_method`: The method to use for upsampling. Can be `:nearest` or `:bilinear`.
"""
function UNet(encoder_config::EncoderConfig; inchannels=3, kw...)
    return UNet(
        extract_encoder(build_encoder(encoder_config; inchannels)), 
        encoder_block_dims(encoder_config); 
        kw...
    )
end

function UNet(encoder::Flux.Chain, encoder_dims::NTuple{N,Int}; decoder_dims=(64,128,256,512,1024), batch_norm=true, nclasses=1, upsample_method=:nearest) where N
    # Check Arguments
    @argcheck nclasses > 0

    # Build Decoder and Head
    decoder = UNetDecoder(encoder_dims; decoder_dims, batch_norm, upsample_method)
    head = SegmentationHead(decoder_dims[1], nclasses)
    return UNet(encoder, decoder, head)
end

function (m::UNet)(x::AbstractArray{<:Real,4})
    features = Flux.activations(m.encoder, x)
    decoder_out = m.decoder(features)
    return Flux.upsample_bilinear(decoder_out, size=feature_size(x)) |> m.head
end