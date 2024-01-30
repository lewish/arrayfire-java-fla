package arrayfire;

public record Tileable<T extends DataType<?, ?>, S extends Shape<?, ?, ?, ?>>(Tensor<T, S> tensor) {
}
