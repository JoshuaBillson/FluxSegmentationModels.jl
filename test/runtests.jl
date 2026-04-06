using FluxSegmentationModels
using Test

const IMSIZE = 224
const CHANNELS = 3

@testset "FluxSegmentationModels.jl" begin
    # Encoder Configurations
    RESNETS = [ResNet(depth=d) for d in (18, 34, 50, 101, 152)]
    CONVNEXTS = [ConvNeXt(config=c) for c in (:pico, :tiny, :small, :base, :large, :xlarge)]
    VITS = [ViT(config=c, imsize=(IMSIZE,IMSIZE), pretrain=false) for c in (:tiny, :small, :base, :large, :huge)]
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
