function UNet(encoder::EncoderConfig; decoder_dims=[64,128,256,512], batch_norm=true, inchannels=3, nclasses=1)
    SegmentationModel(
        build_encoder(encoder; inchannels),
        UNetDecoder(encoder_feature_dims(encoder), encoder_feature_scales(encoder); decoder_dims, batch_norm), 
        SegmentationHead(decoder_dims[1], nclasses, encoder_feature_scales(encoder)[1])
    )
end