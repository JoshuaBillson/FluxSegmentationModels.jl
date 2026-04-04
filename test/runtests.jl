using FluxSegmentationModels
using Test

function generate_features(encoder_dims, encoder_scales; imsize=256)
    return tuple([rand(Float32, imsize ÷ scale, imsize ÷ scale, dim, 1) for (dim, scale) in zip(encoder_dims, encoder_scales)]...)
end

const IMSIZE = 256
const CHANNELS = 3

const UNET_ENCODER_DIMS = [64,128,256,512,1024]
const UNET_ENCODER_SCALES = [1,2,4,8,16]

const RESNET50_ENCODER_DIMS = [256,512,1024,2048]
const RESNET50_ENCODER_SCALES = [4,8,16,32]

@testset "FluxSegmentationModels.jl" begin
    # Encoder Configurations
    RESNETS = [ResNet(depth=d) for d in (18, 34, 50, 101, 152)]
    CONVNEXTS = [ConvNeXt(config=c) for c in (:pico, :tiny, :small, :base, :large, :xlarge)]
    VITS = [ViT(config=c, imsize=(IMSIZE,IMSIZE)) for c in (:tiny, :small, :base, :large, :huge)]
    ENCODERS = vcat(RESNETS, CONVNEXTS)

    # Generate Dummy Input
    INPUT = rand(Float32, IMSIZE, IMSIZE, CHANNELS, 1)

    # Test UNet
    for encoder in ENCODERS
        model = UNet(encoder; inchannels=CHANNELS, nclasses=10)
        @test size(model(INPUT)) == (IMSIZE, IMSIZE, 10, 1)
    end

    # Test FPN
    for encoder in ENCODERS
        model = FPN(encoder; inchannels=CHANNELS, nclasses=10)
        @test size(model(INPUT)) == (IMSIZE, IMSIZE, 10, 1)
    end

    # Test SegFormer
    for encoder in ENCODERS
        model = SegFormer(encoder; inchannels=CHANNELS, nclasses=10)
        @test size(model(INPUT)) == (IMSIZE, IMSIZE, 10, 1)
    end

    # Test SETR
    for encoder in VITS
        model = SETR(encoder; inchannels=CHANNELS, nclasses=10)
        @test size(model(INPUT)) == (IMSIZE, IMSIZE, 10, 1)
    end
end
