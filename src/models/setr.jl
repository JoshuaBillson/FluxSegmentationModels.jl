"""

    SETR(encoder_config::EncoderConfig; inchannels=3, kw...)
    SETR(encoder, encoder_dims; encoder_blocks=(3,6,9,12), patchsize=(16,16), batch_norm=true, nclasses=1)

Construct a SETR style segmentation model.

# Parameters
- `encoder_config`: An `EncoderConfig` object specifying the architecture and configuration of the encoder to be built and used in the SETR.
- `encoder`: A `Flux.Chain` layer containing the blocks of the encoder to be used in the SETR.
- `encoder_dims`: A tuple containing the feature dimension of each encoder block output ordered from first to last.
- `encoder_blocks`: A tuple containing the indices of the encoder blocks to be used in the decoder ordered from first to last. Default is `(3,6,9,12)`.
- `patchsize`: The patch size to use for the input to the encoder. Default is `(16,16)`.
- `batch_norm`: If true, a batch norm operation will be applied after each convolution in the decoder. Default is `true`.
- `nclasses`: The number of output classes for the segmentation task. Default is `1`.
"""
struct SETR{E,D,H}
    patchsize::NTuple{2,Int}
    encoder_block_indices::NTuple{4,Int}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand SETR

function SETR(encoder_config::EncoderConfig; inchannels=3, kw...)
    return SETR(
        extract_encoder(build_encoder(encoder_config; inchannels)), 
        encoder_block_dims(encoder_config);
        kw...
    )
end

function SETR(encoder, encoder_dims; encoder_blocks=(3,6,9,12), patchsize=(16,16), batch_norm=true, nclasses=1)
    @argcheck all(encoder_blocks .> 0)
    @argcheck issorted(encoder_blocks)
    @argcheck all(patchsize .> 0)
    @argcheck nclasses > 0

    return SETR(
        patchsize, 
        encoder_blocks,
        encoder,
        SETRDecoder(encoder_dims[end]; n_features=length(encoder_blocks), batch_norm),
        SegmentationHead(encoder_dims[end], nclasses)
    )
end

function (m::SETR)(x)
    imsize = feature_size(x)
    featsize = div.(imsize, m.patchsize)
    encoder_out = Flux.activations(m.encoder, x)
    feature_maps = map(x -> seq2img(x, featsize), index_tuple(encoder_out, reverse(m.encoder_block_indices)))
    decoder_out = m.decoder(feature_maps)
    return Flux.upsample_bilinear(decoder_out, size=imsize) |> m.head
end