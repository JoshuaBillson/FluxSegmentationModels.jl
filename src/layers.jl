"""
    ConvBlock(kernel_size::Tuple{Int,Int}, in::Int, out::Int; act=Flux.relu, depth=2, norm=:BN)

A block of convolutional layers with optional batch normalization.

# Parameters
- `kernel_size`: The size of the filter to use for each layer.
- `in`: The number of input features.
- `out`: The number of output features.
- `act`: The activation function to apply after each convolution.
- `depth`: The number of successive convolutions in the block.
- `norm`: Specifies the type of layer normalization. One of `:BN`, `:LN`, or `nothing`
"""
function ConvBlock(kernel_size::Tuple{Int,Int}, in::Int, out::Int; act=Flux.relu, depth=2, norm=:BN)
    return Flux.Chain([Conv(kernel_size, i == 1 ? in : out, out; act, norm) for i in 1:depth]...)
end

"""
    Conv(kernel_size::Tuple{Int,Int}, in::Int, out::Int; act=Flux.relu, norm=:BN, groups=1, dilation=1)

A single convolutional layer with optional batch normalization.

# Parameters
- `kernel_size`: The size of the filter to use for each layer.
- `in`: The number of input features.
- `out`: The number of output features.
- `act`: The activation function to apply after each convolution.
- `norm`: Specifies the type of layer normalization. One of `:BN`, `:LN`, or `nothing`.
- `group`: The number of groups to divide the convolution into.
- `dilation`: The dilation rate for the kernel.
"""
function Conv(kernel_size::Tuple{Int,Int}, in::Int, out::Int; act=Flux.relu, norm=:BN, groups=1, dilation=1)
    @match norm begin
        :BN => Flux.Chain(
            Flux.Conv(kernel_size, in=>out; groups, dilation, pad=Flux.SamePad(), bias=false),
            Flux.BatchNorm(out, act))
        :LN => Flux.Chain(
            Flux.Conv(kernel_size, in=>out; groups, dilation, pad=Flux.SamePad(), bias=false),
            Base.Fix2(permutedims, (3,1,2,4)), 
            Flux.LayerNorm(out, act), 
            Base.Fix2(permutedims, (2,3,1,4)))
        nothing => Flux.Conv(kernel_size, in=>out, act; groups, dilation, pad=Flux.SamePad())
    end
end

"""
    SeparableConv(kernel::Tuple{Int,Int}, in::Int, out::Int; stride=1, act=identity, pad=Flux.SamePad(), norm=:BN)

A single separable convolutional layer with optional batch normalization.

# Parameters
- `kernel_size`: The size of the filter to use for each layer.
- `in`: The number of input features.
- `out`: The number of output features.
- `stride`: The stride of the convolution operation.
- `act`: The activation function to apply after each convolution.
- `norm`: Specifies the type of layer normalization. One of `:BN`, `:LN`, or `nothing`.
"""
function SeparableConv(kernel::Tuple{Int,Int}, in::Int, out::Int; stride=1, act=identity, norm=:BN)
    @match norm begin
        :BN => Flux.Chain(
            Flux.DepthwiseConv(kernel, in=>in; pad=Flux.SamePad(), stride), 
            Flux.BatchNorm(in), 
            Flux.Conv((1,1), in=>out, act, pad=Flux.SamePad()))
        :LN => Flux.Chain(
            Flux.DepthwiseConv(kernel, in=>in; pad=Flux.SamePad(), stride), 
            Base.Fix2(permutedims, (3,1,2,4)),
            Flux.LayerNorm(in), 
            Base.Fix2(permutedims, (2,3,1,4)),
            Flux.Conv((1,1), in=>out, act, pad=Flux.SamePad()))
        nothing => Flux.Chain(
            Flux.DepthwiseConv(kernel, in=>in; pad=Flux.SamePad(), stride), 
            Flux.Conv((1,1), in=>out, act, pad=Flux.SamePad()))
    end
end


"""
    SegmentationHead(dims::Int, nclasses::Int)

Construct a segmentation head consisting of a 1x1 convolution followed by an activation function.

# Parameters
- `dims`: The number of input features to the segmentation head.
- `nclasses`: The number of output classes for the segmentation task.
""" 
function SegmentationHead(dims::Int, nclasses::Int)
    @argcheck nclasses >= 1
    @argcheck dims >= 1
    activation = nclasses == 1 ? Flux.σ : Flux.softmax $ (;dims=3)
    return Flux.Chain(Flux.Conv((1,1), dims => nclasses), activation)
end

struct UpsampleNearest
    factor::Int
end

Flux.@layer :expand UpsampleNearest

function (m::UpsampleNearest)(x::AbstractArray{<:Real,N}) where {N} 
    return Flux.upsample_nearest(x, size=feature_size(x) .* m.factor)
end

struct UpsampleBilinear
    factor::Int
end

Flux.@layer :expand UpsampleBilinear

function (m::UpsampleBilinear)(x::AbstractArray{<:Real,N}) where {N} 
    return Flux.upsample_bilinear(x, size=feature_size(x) .* m.factor)
end

"""
    Upsample(factor::Int; method=:nearest)

Construct a layer that takes an image with shape WHCN and upsamples it by `factor`.

# Parameters
- `factor`: The factor by which to upscale the input image.
- `method`: The type of resampling to use; must be one of `:nearest` or `:bilinear`.
"""
function Upsample(factor::Int; method=:nearest)
    @argcheck factor >= 1
    @argcheck method in (:nearest, :bilinear)
    @match method begin
        :nearest => UpsampleNearest(factor)
        :bilinear => UpsampleBilinear(factor)
    end
end

"""
    FeaturePyramid(blocks...)

Construct a feature pyramid from a tuple of convolutional blocks. The output of each block is upsampled and added to the output of the next block.
"""
function FeaturePyramid(blocks...; upsample_method=:nearest)
    return Flux.Parallel(
        accumulate_features $ (up_and_add $ (;method=upsample_method)),
        blocks...,
    )
end

"""
    LeftFold(f, blocks...)

Construct a layer that applies a left fold over the input features using the provided function and blocks. The first block is applied to the first two inputs, then the second block is applied to the output of the first block and the third input, and so on.
"""
struct LeftFold{F<:Function,B<:Tuple}
    f::F
    blocks::B
end

LeftFold(f::Function, blocks...) = LeftFold(f, blocks)

Flux.@layer :expand LeftFold

function (m::LeftFold)(xs::NTuple{N,<:AbstractArray{<:Real,4}}) where N
    return _lfold(m.f, m.blocks, xs)
end

@generated function _lfold(f::Function, layers::NTuple, xs::NTuple{N,<:AbstractArray}) where {N}
    accs = [Symbol(:acc_, i) for i in 1:N]
    vals = [Symbol(:val_, i) for i in 1:N-1]
    exprs = Expr[]

    # Initialize Accumulator
    push!(exprs, :($(accs[1]) = xs[1]))

    # Forward Pass through Decoder Blocks
    for i in 2:N
        push!(exprs, :($(vals[i-1]) = xs[$i]))
        push!(exprs, :($(accs[i]) = layers[$(i-1)](f($(accs[i-1]), $(vals[i-1])))))
    end

    push!(exprs, :(return $(accs[N])))

    return Expr(:block, exprs...)
end

@generated function _rfold(f::Function, layers::NTuple, xs::NTuple{N,<:AbstractArray}) where {N}
    accs = [Symbol(:acc_, i) for i in 1:N]
    vals = [Symbol(:val_, i) for i in 1:N-1]
    exprs = Expr[]

    # Initialize Accumulator
    push!(exprs, :($(accs[1]) = xs[N]))

    # Forward Pass through Decoder Blocks
    for i in 1:N-1
        push!(exprs, :($(vals[i]) = xs[$(N-i)]))
        push!(exprs, :($(accs[i+1]) = layers[$i](f($(accs[i]), $(vals[i])))))
    end

    push!(exprs, :(return $(accs[N])))

    return Expr(:block, exprs...)
end

accumulate_features(f::Function, xs...) = accumulate_features(f, xs)
@generated function accumulate_features(f::Function, xs::NTuple{N,<:AbstractArray{<:Real,4}}) where {N}
    # symbols for intermediate outputs
    _xs = [Symbol(:x, i) for i in 1:N]

    # Build expression for each step of accumulation
    exprs = Expr[]
    push!(exprs, :($(_xs[1]) = xs[1]))
    for i in 2:N
        push!(exprs, :($(_xs[i]) = f($(_xs[i-1]), xs[$i])))
    end

    # return (x1, x2, ..., xN)
    push!(exprs, :(return ($( _xs... ),)))

    # return the full expression block
    return Expr(:block, exprs...)
end

aggregate_features(f::Function, xs...) = aggregate_features(f, xs)
function aggregate_features(f::Function, xs::NTuple{N,<:AbstractArray{<:Real,4}}) where N
    return foldl(f, xs)
end

@generated function chunk(x::AbstractArray{<:Real,3}, nchunks::Val{N}) where N
    exprs = []
    chunks = [Symbol(:chunk, i) for i in 1:N]
    push!(exprs, :(chunksize = size(x, 2) ÷ $N))
    for i in 1:N
        push!(exprs, :($(chunks[i]) = x[:, ($(i-1) * chunksize + 1):($(i) * chunksize), :]))
    end
    push!(exprs, :(return ($( chunks... ),)))
    return Expr(:block, exprs...)
end