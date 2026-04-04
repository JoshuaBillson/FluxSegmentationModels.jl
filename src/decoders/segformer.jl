"""
    SegFormerDecoder(encoder_dims::NTuple{N,Int}; embed_dim=768, dropout=0.0)

A MLP-style decoder as used in SegFormer. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `embed_dim`: The size of the embedding dimension to use. Shared by all layers.
- `dropout`: The dropout probability to use after the last layer.

# Input
A tuple of WHCN arrays corresponding to the output of each block of a matching encoder ordered from first to last.
"""
struct SegFormerDecoder{M,F}
    mlps::M
    fuse::F
end

Flux.@layer :expand SegFormerDecoder

function SegFormerDecoder(encoder_dims::NTuple{N,Int}; embed_dim=768, dropout=0.0) where N
    mlps = Flux.Parallel(
        concatenate_features ∘ upsample_features,
        [Flux.Conv((1,1), dim=>embed_dim) for dim in encoder_dims]...
    )

    fuse = Flux.Chain(
        Flux.Conv((1,1), embed_dim*N=>embed_dim), 
        Flux.BatchNorm(embed_dim, Flux.relu), 
        Flux.Dropout(dropout)
    )

    return SegFormerDecoder(mlps, fuse)
end

function (m::SegFormerDecoder)(xs::NTuple{N,<:AbstractArray{<:Real,4}}) where N
    return m.mlps(xs) |> m.fuse
end