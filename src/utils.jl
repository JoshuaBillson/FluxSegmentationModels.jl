feature_size(x::AbstractArray{<:Any,N}) where N = size(x)[1:N-2]

reverse_features(xs::NTuple{N}) where N = xs[N:-1:1]

up_and_add_nearest(x::AbstractArray, skip::AbstractArray) = up_and_add(x, skip; method=:nearest)

up_and_add_bilinear(x::AbstractArray, skip::AbstractArray) = up_and_add(x, skip; method=:bilinear)

function up_and_add(x::AbstractArray{<:Real,N}, skip::AbstractArray{<:Real,N}; method=:nearest) where N
    up_size = feature_size(skip)
    if feature_size(x) != up_size
        return upsample_features(x, up_size; method) .+ skip
    end
    return x .+ skip
end

up_and_concat_nearest(x::AbstractArray, skip::AbstractArray) = up_and_concat(x, skip; method=:nearest)

up_and_concat_bilinear(x::AbstractArray, skip::AbstractArray) = up_and_concat(x, skip; method=:bilinear)

function up_and_concat(x::AbstractArray{<:Any,N}, skip::AbstractArray{<:Any,N}; method=:nearest) where N
    up_size = feature_size(skip)
    if feature_size(x) != up_size
        return cat(upsample_features(x, up_size; method), skip; dims=N-1)
    end
    return cat(x, skip; dims=N-1)
end

upsample_features(xs...; kw...) = upsample_features(xs; kw...)
function upsample_features(xs::NTuple{N,<:AbstractArray{<:Real,N}}; kw...) where N
    up_size = max_feature_size(xs)
    return map(x -> upsample_features(x, up_size; kw...), xs)
end
function upsample_features(x::AbstractArray{<:Real,N}, scale::Int; method=:nearest) where {N}
    return upsample_features(x, feature_size(x) .* scale; method)
end
function upsample_features(x::AbstractArray{<:Real,N}, size::NTuple{S,Int}; method=:nearest) where {N,S}
    size == feature_size(x) && return x  # No need to upsample if the size is already correct
    @match method begin
        :nearest => Flux.upsample_nearest(x, size=size)
        :bilinear => Flux.upsample_bilinear(x, size=size)
        _ => error("Unsupported upsampling method: $method")
    end
end

concatenate_features_nearest(xs...) = concatenate_features(xs; method=:nearest)
concatenate_features_nearest(xs::Tuple) = concatenate_features(xs; method=:nearest)

concatenate_features_bilinear(xs...) = concatenate_features(xs; method=:bilinear)
concatenate_features_bilinear(xs::Tuple) = concatenate_features(xs; method=:bilinear)

concatenate_features(xs...; kw...) = concatenate_features(xs; kw...)
function concatenate_features(xs::NTuple{N,<:AbstractArray{<:Real,D}}; method=:nearest) where {N,D}
    return cat(xs..., dims=D-1)
    up_size = max_feature_size(xs)
    return cat(map(x -> upsample_features(x, up_size; method), xs)...; dims=D-1)
end

add_features(xs...; kw...) = add_features(xs; kw...)
function add_features(xs::NTuple{N,<:AbstractArray{<:Real,D}}; method=:nearest) where {N,D}
    up_size = max_feature_size(xs)
    return +(map(x -> upsample_features(x, up_size; method), xs)...)
end

max_feature_size(xs...) = max_feature_size(xs)
function max_feature_size(xs::NTuple{N,<:AbstractArray}) where N
    return maximum(map(feature_size, xs))
end

function seq2img(x::AbstractArray{<:Real,3}, imsize::NTuple{N,Int}) where N
    C, _, B = size(x)
    newsize = (C, imsize..., B)
    permutation = (2:2+N-1..., 1, N+2)
    return @pipe _strip_tokens(x, imsize) |> reshape(_, newsize) |> permutedims(_, permutation)
end

@generated function whcn_2_cwhn(x::AbstractArray{<:Any,N}) where N
    D = N - 2
    permutation = (D+1, 1:D..., D+2)
    return :(permutedims(x, ($(permutation...),)))
end

@generated function cwhn_2_whcn(x::AbstractArray{<:Any,N}) where N
    D = N - 2
    permutation = (2:D+1..., 1, N)
    return :(permutedims(x, ($(permutation...),)))
end

function _strip_tokens(x::AbstractArray{<:Real,3}, imsize::NTuple{N,Int}) where N
    L = size(x,2)
    n_tokens = L - prod(imsize)
    return x[:, n_tokens+1:end, :]
end

@generated function index_tuple(features::NTuple{N}, indices::NTuple{M,Int}) where {N,M}
    exprs = []
    feats = [Symbol(:x, i) for i in 1:M]
    for i in 1:M
        push!(exprs, :($(feats[i]) = features[indices[$i]]))
    end
    push!(exprs, :(return ($( feats... ),)))
    return Expr(:block, exprs...)
end