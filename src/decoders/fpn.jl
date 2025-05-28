
"""
    FPNDecoder(encoder_dims::AbstractVector{<:Integer}, encoder_scales::AbstractVector{<:Integer}; embed_dim=256, dropout=0.0, batch_norm=true)

A MLP-style decoder as used in SegFormer. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `encoder_scales`: The scale of each encoder block output with respect to the input ordered from first to last.
- `embed_dim`: The size of the embedding dimension to use. Shared by all layers.
- `dropout`: The dropout probability to use after the last layer.
- `batch_norm`: Use batch normalization after each convolution.
"""
function FPNDecoder(encoder_dims::AbstractVector{<:Integer}, encoder_scales::AbstractVector{<:Integer}; embed_dim=256, dropout=0.0, batch_norm=true)
    # Validate Arguments
    @argcheck 0.0 <= dropout <= 1.0
    @argcheck length(encoder_dims) == length(encoder_scales)
    @argcheck all(==(2), encoder_scales[2:end] .÷ encoder_scales[1:end-1])

    # Construct FPN Blocks
    fpn_blocks = Any[]
    for encoder_dim in encoder_dims
        push!(fpn_blocks, Flux.Conv((1,1), encoder_dim=>embed_dim))
    end

    # Construct Segmentation Heads
    norm = batch_norm ? :BN : nothing
    seg_blocks = Any[ConvBlock((3,3), embed_dim, embed_dim; norm)]
    upsamples = encoder_scales .÷ minimum(encoder_scales)
    for upsample in upsamples[2:end]
        push!(
            seg_blocks, 
            Flux.Chain(ConvBlock((3,3), embed_dim, embed_dim; norm)..., Upsample(upsample; method=:nearest))
        )
    end

    # Assemble Decoder
    Flux.Chain(
        reverse,
        Flux.Parallel(up_and_add, reverse(fpn_blocks)...),
        Flux.Parallel((xs...) -> cat(xs..., dims=3), reverse(seg_blocks)...),
        Flux.Chain(Flux.Dropout(dropout), Conv((3,3), length(encoder_dims)*embed_dim, embed_dim))
    )
end

up_and_add(features...) = accumulate((x, skip) -> Flux.upsample_bilinear(x, (2,2)) .+ skip, features)