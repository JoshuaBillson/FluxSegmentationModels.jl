using FluxSegmentationModels
using Test

function generate_features(encoder_dims, encoder_scales; imsize=256)
    return tuple([rand(Float32, imsize ÷ scale, imsize ÷ scale, dim, 1) for (dim, scale) in zip(encoder_dims, encoder_scales)]...)
end

const IMSIZE = 256

const UNET_ENCODER_DIMS = [64,128,256,512,1024]
const UNET_ENCODER_SCALES = [1,2,4,8,16]

const RESNET50_ENCODER_DIMS = [256,512,1024,2048]
const RESNET50_ENCODER_SCALES = [4,8,16,32]

@testset "FluxSegmentationModels.jl" begin
    # Generate Encoder Features
    unet_features = generate_features(UNET_ENCODER_DIMS, UNET_ENCODER_SCALES; imsize=IMSIZE)
    resnet_features = generate_features(RESNET50_ENCODER_DIMS, RESNET50_ENCODER_SCALES; imsize=IMSIZE)

    # Test UNet Decoder
    standard_unet_decoder = UNetDecoder(UNET_ENCODER_DIMS, UNET_ENCODER_SCALES)
    resnet_unet_decoder = UNetDecoder(RESNET50_ENCODER_DIMS, RESNET50_ENCODER_SCALES)
    @test size(standard_unet_decoder(unet_features)) == (IMSIZE,IMSIZE,64,1)
    @test size(resnet_unet_decoder(resnet_features)) == (IMSIZE÷4,IMSIZE÷4,64,1)

    # Test FPN Decoder
    standard_fpn_decoder = FPNDecoder(UNET_ENCODER_DIMS, UNET_ENCODER_SCALES)
    resnet_fpn_decoder = FPNDecoder(RESNET50_ENCODER_DIMS, RESNET50_ENCODER_SCALES)
    @test size(standard_fpn_decoder(unet_features)) == (IMSIZE,IMSIZE,256,1)
    @test size(resnet_unet_decoder(resnet_features)) == (IMSIZE÷4,IMSIZE÷4,64,1)
end
