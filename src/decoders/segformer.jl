"""
    SegFormerDecoder(feature_dims::Vector{Int}, feature_scales::Vector{Int}; embed_dim=768, dropout=0.0, norm=:BN)

A MLP-style decoder as used in SegFormer. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `encoder_scales`: The scale of each encoder block output with respect to the input ordered from first to last.
- `embed_dim`: The size of the embedding dimension to use. Shared by all layers.
- `dropout`: The dropout probability to use after the last layer.
- `norm`: One of either `:BN` or `:LN`.
"""
function SegFormerDecoder(encoder_dims::AbstractVector{<:Integer}, encoder_scales::AbstractVector{<:Integer}; embed_dim=768, dropout=0.0, norm=:BN)
    @argcheck norm in (:LN, :BN)
    @argcheck 0.0 <= dropout <= 1.0
    @argcheck length(encoder_dims) == length(encoder_scales)
    upsamples = encoder_scales .÷ minimum(encoder_scales)
    Flux.Chain(
        Flux.Parallel(
            (x...) -> cat(x...; dims=3), 
            [_segformer_mlp(dim, embed_dim, upsample) for (dim, upsample) in zip(encoder_dims, upsamples)]...
        ), 
        _segformer_fuse(embed_dim, length(encoder_dims), norm, dropout)
    )
end

function _segformer_mlp(indim::Int, outdim::Int, upsample::Int)
    if upsample > 1
        return Flux.Chain(Flux.Conv((1,1), indim=>outdim), Upsample(upsample; method=:bilinear))
    end
    return Flux.Conv((1,1), indim=>outdim)
end

function _segformer_fuse(embed_dim::Int, width::Int, norm::Symbol, dropout::Float64)
    @match norm begin
        :LN => Flux.Chain(
            Base.Fix2(permutedims, (3,1,2,4)),
            Flux.Dense(embed_dim*width=>embed_dim),
            Flux.LayerNorm(embed_dim, Flux.relu), 
            Base.Fix2(permutedims, (2,3,1,4)),
            Flux.Dropout(dropout))
        :BN => Flux.Chain(
            Flux.Conv((1,1), embed_dim*width=>embed_dim), 
            Flux.BatchNorm(embed_dim, Flux.relu), 
            Flux.Dropout(dropout))
    end
end