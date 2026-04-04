"""
    UNetDecoder(encoder_dims::NTuple{N,Int}; decoder_dims=(64,128,256,512,1024), batch_norm=true, upsample_method=:nearest)

Construct a U-Net style decoder that expects a tuple of WHCN arrays corresponding to the output
of each block of a matching encoder.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `decoder_dims`: The feature dimension of each decoder block ordered from top to bottom.
- `batch_norm`: If true, a batch norm operation will be applied after each convolution in the decoder.
- `upsample_method`: The method to use for upsampling. Can be `:nearest` or `:bilinear`.
"""
struct UNetDecoder{B}
    blocks::B
end

Flux.@layer :expand UNetDecoder

function UNetDecoder(encoder_dims::NTuple{N,Int}; decoder_dims=(64,128,256,512,1024), batch_norm=true, upsample_method=:nearest) where N
    # Check Arguments
    @argcheck all(encoder_dims .>= 1)
    @argcheck all(decoder_dims .>= 1)
    @argcheck length(encoder_dims) <= length(decoder_dims)
    @argcheck upsample_method in (:nearest, :bilinear)

    # Create Decoder Blocks
    decoder_out_dims = decoder_dims[1:N]
    decoder_in_dims = encoder_dims .+ (decoder_out_dims[2:end]..., 0)
    decoder_blocks = map(decoder_in_dims, decoder_out_dims) do in_dim, out_dim
        ConvBlock((3,3), in_dim, out_dim; norm=(batch_norm ? :BN : nothing))
    end

    # Assemble Decoder
    decoder = Flux.PairwiseFusion(
        upsample_method == :nearest ? up_and_concat_nearest : up_and_concat_bilinear,
        reverse(decoder_blocks)...,
    )

    return UNetDecoder(decoder)
end

function (d::UNetDecoder)(xs::NTuple{N, <:AbstractArray{<:Real,N}}) where N
    return reverse(xs) |> d.blocks |> last
end