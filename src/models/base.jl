struct SegmentationModel{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand SegmentationModel

(m::SegmentationModel)(x) = Flux.activations(m.encoder, x) |> m.decoder |> m.head

function SegmentationHead(in_features::Integer, nclasses::Integer, upsample::Integer)
    @argcheck nclasses >= 1
    @argcheck in_features >= 1
    @argcheck upsample >= 1
    if upsample > 1
        return Flux.Chain(
            Upsample(upsample; method=:bilinear),
            Flux.Conv((3,3), in_features=>in_features, Flux.relu, pad=Flux.SamePad()),
            Flux.Conv((1,1), in_features=>nclasses),
            ClassificationFunction(nclasses)
        )
    else
        return Flux.Chain(
            Flux.Conv((1,1), in_features=>nclasses), 
            ClassificationFunction(nclasses)
        )
    end
end

function ClassificationFunction(nclasses::Int)
    if nclasses > 1
        return x -> Flux.softmax(x; dims=3)
    else
        return x -> Flux.sigmoid.(x)
    end
end