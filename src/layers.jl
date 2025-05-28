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

A block of convolutional layers with optional batch normalization.

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
            Flux.Conv(kernel_size, in=>out; groups, dilation, pad=Flux.SamePad()),
            Flux.BatchNorm(out, act),)
        :LN => Flux.Chain(
            Flux.Conv(kernel_size, in=>out; groups, dilation, pad=Flux.SamePad()),
            Base.Fix2(permutedims, (3,1,2,4)), 
            Flux.LayerNorm(out, act), 
            Base.Fix2(permutedims, (2,3,1,4)))
        nothing => Flux.Conv(kernel_size, in=>out, act; groups, dilation, pad=Flux.SamePad())
    end
end

"""
    SeparableConv(kernel::Tuple{Int,Int}, in::Int, out::Int; stride=1, act=identity, pad=Flux.SamePad(), norm=:BN)

A block of convolutional layers with optional batch normalization.

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
        :nearest => Base.Fix2(Flux.upsample_nearest, (factor,factor))
        :bilinear => Base.Fix2(Flux.upsample_bilinear, (factor,factor))
    end
end