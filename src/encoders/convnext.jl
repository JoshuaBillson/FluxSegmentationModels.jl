"""
    ConvNeXt(;config=:tiny)

A ConvNeXt style encoder.

# Parameters
- `config`: The size of the ConvNeXt model to use. Can be `:pico`, `:tiny`, `:small`, `:base`, `:large`, or `:xlarge`.
"""
struct ConvNeXt <: EncoderConfig
    config::Symbol
end

function ConvNeXt(;config=:tiny)
    @argcheck config in (:pico,:tiny,:small,:base,:large,:xlarge)
    return ConvNeXt(config)
end

encoder_block_dims(x::ConvNeXt) = _convnext_filters(x.config)

function build_encoder(x::ConvNeXt; inchannels=3, nclasses=1000)
    model = _build_convnext(x.config, inchannels, nclasses)
    head = model.layers[2]
    encoder = model.layers[1]
    encoder_stem = encoder[1:2]
    encoder_body = @match x.config begin
        :pico => Flux.Chain(
            encoder[3:5], 
            encoder[6:9], 
            encoder[10:17], 
            encoder[18:20] )
        :tiny => Flux.Chain(
            encoder[3:6], 
            encoder[7:11], 
            encoder[12:22], 
            encoder[23:26] )
        :small || :base || :large || :xlarge => Flux.Chain(
            encoder[3:6], 
            encoder[7:11], 
            encoder[12:40], 
            encoder[41:44] )
    end

    return Flux.Chain(
        encoder = Flux.Chain(
            stem = encoder_stem, 
            body = encoder_body
        ),
        head = head
    )
end

function _build_convnext(config::Symbol, inchannels::Int, nclasses::Int)
    _filters = collect(_convnext_filters(config))
    @match config begin
        :pico => Metalhead.build_convnext([2, 2, 6, 2], _filters; inchannels, nclasses)
        :tiny => Metalhead.build_convnext([3, 3, 9, 3], _filters; inchannels, nclasses)
        :small || :base || :large || :xlarge => Metalhead.build_convnext([3, 3, 27, 3], _filters; inchannels, nclasses)
    end
end

function _convnext_filters(config::Symbol)
    @match config begin
        :pico => (64, 128, 256, 512)
        :tiny || :small => (96,192,384,768)
        :base => (128,256,512,1024)
        :large => (192,384,768,1536)
        :xlarge => (256,512,1024,2048)
    end
end