"""
    UNetDecoder(encoder_dims, encoder_scales; decoder_dims=[64,128,256,512], batch_norm=true)

Construct a U-Net style decoder that expects a tuple of WHCN arrays corresponding to the output
of each block of a matching encoder.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `encoder_scales`: The scale of each encoder block output with respect to the input ordered from first to last.
- `decoder_dims`: The feature dimension of each decoder block ordered from top to bottom.
- `batch_norm`: If true, a batch norm operation will be applied after each convolution in the decoder.
"""
function UNetDecoder(encoder_dims::AbstractVector{<:Integer}, encoder_scales::AbstractVector{<:Integer}; decoder_dims=[64,128,256,512], batch_norm=true)
    # Check Arguments
    @argcheck all(x -> x >= 1, encoder_dims)
    @argcheck all(x -> x >= 1, encoder_scales)
    @argcheck all(x -> x >= 1, decoder_dims)
    @argcheck length(encoder_dims) == length(encoder_scales)
    @argcheck length(encoder_dims) <= length(decoder_dims) + 1

    # Create Decoder Blocks
    decoder_blocks = Any[]
    skip_dims = encoder_dims[1:end-1]
    up_dims = vcat(decoder_dims[2:end], encoder_dims[end])
    up_factors = vcat([1], encoder_scales[2:end] .÷ encoder_scales[1:end-1])
    for (up_dim, skip_dim, up_factor, decoder_dim) in zip(up_dims, skip_dims, up_factors, decoder_dims)
        if up_factor == 1
            push!(
                decoder_blocks, 
                Flux.Chain(
                    ConvBlock((3,3), up_dim+skip_dim, decoder_dim; norm=(batch_norm ? :BN : nothing))...,
                )
            )
        else
            push!(
                decoder_blocks, 
                Flux.Chain(
                    ConvBlock((3,3), up_dim+skip_dim, decoder_dim; norm=(batch_norm ? :BN : nothing))...,
                    Upsample(up_factor; method=:nearest),
                )
            )
        end
    end

    # Assemble Decoder from Blocks
    Flux.Chain(
        reverse,
        Flux.PairwiseFusion(
            (a, b) -> cat(a, b; dims=3),
            Upsample(up_factors[end]; method=:nearest),
            reverse(decoder_blocks)...
        ),
        last
    )
end
