"""
    FPNDecoder(encoder_dims::NTuple{N,Int}; pyramid_dim=256, segmentation_dim=128, 
               upsample_method=:nearest, merge_policy=:add, dropout=0.0)

A Feature Pyramid Network style decoder. Expects a tuple of block-wise activations as input.

# Parameters
- `encoder_dims`: The feature dimension of each encoder block output ordered from first to last.
- `pyramid_dim`: The size of the embedding dimension to use in the feature pyramid.
- `segmentation_dim`: The size of the segmentation dimension to use in the segmentation blocks.
- `upsample_method`: The method to use for upsampling. Can be `:nearest` or `:bilinear`.
- `merge_policy`: The method to use for merging features in the segmentation blocks. Can be `:add` or `:concat`.
- `dropout`: The dropout probability to use after the last layer.

# Example
```julia
julia> x = rand(Float32, 256, 256, 3, 1);

julia> encoder = FluxSegmentationModels.build_encoder(ResNet(depth=18));

julia> decoder = FluxSegmentationModels.FPNDecoder((64,128,256,512));

julia> features = Flux.activations(encoder, x);

julia> size.(features)
((64, 64, 64, 1), (32, 32, 128, 1), (16, 16, 256, 1), (8, 8, 512, 1))

julia> decoder_out = decoder(features);

julia> size(decoder_out)
(64, 64, 256, 1)
```
"""
struct FPNDecoder{P,S,F}
    feature_pyramid::P
    seg_blocks::S
    fuse::F
end

Flux.@layer :expand FPNDecoder

function FPNDecoder(
    encoder_dims::NTuple{N,Int}; pyramid_dim=256, segmentation_dim=128, 
    upsample_method=:nearest, merge_policy=:add, dropout=0.0) where N

    # Validate Arguments
    @argcheck 0.0 <= dropout <= 1.0
    @argcheck all(encoder_dims .>= 1)
    @argcheck pyramid_dim > 0
    @argcheck segmentation_dim > 0
    @argcheck upsample_method in (:nearest, :bilinear)
    @argcheck merge_policy in (:add, :concat)

    # Construct FPN Blocks
    feature_pyramid = Flux.Parallel(
        accumulate_features $ (upsample_method == :nearest ? up_and_add_nearest : up_and_add_bilinear),
        [Flux.Conv((1,1), encoder_dim=>pyramid_dim) for encoder_dim in reverse(encoder_dims)]...
    )

    # Construct Segmentation Heads
    seg_blocks = Flux.Parallel(
        merge_policy == :add ? add_features : concatenate_features,
        [_fpn_seg_block(pyramid_dim, segmentation_dim, N-i) for i in 1:N]...,
    )

    # Construct Feature Fusion Layer
    fusion_dim_in = merge_policy == :add ? segmentation_dim : segmentation_dim * N
    fuse = Flux.Chain(
        Flux.Conv((3,3), fusion_dim_in=>segmentation_dim, pad=Flux.SamePad(), bias=false), 
        Flux.BatchNorm(segmentation_dim, Flux.relu),
        Flux.Dropout(dropout), 
    )

    # Assemble Model
    return FPNDecoder(feature_pyramid, seg_blocks, fuse)
end

function (m::FPNDecoder)(xs::NTuple{N,<:AbstractArray{<:Real,4}}) where N
    return @pipe reverse(xs) |> m.feature_pyramid |> m.seg_blocks |> m.fuse
end

function _fpn_seg_block(indim::Int, outdim::Int, n_upsamples::Int)
    blocks = Any[Conv((3,3), indim, outdim)]
    n_upsamples >= 1 && push!(blocks, Base.Fix2(Flux.upsample_bilinear, (2,2)))
    for _ in 2:n_upsamples
        push!(blocks, Conv((3,3), outdim, outdim))
        push!(blocks, Base.Fix2(Flux.upsample_bilinear, (2,2)))
    end
    return Flux.Chain(blocks...)
end