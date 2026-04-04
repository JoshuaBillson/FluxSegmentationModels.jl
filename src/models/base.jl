struct SegmentationModel{E,D,H}
    encoder::E
    decoder::D
    head::H
end

Flux.@layer :expand SegmentationModel

(m::SegmentationModel)(x) = Flux.activations(m.encoder, x) |> m.decoder |> m.head