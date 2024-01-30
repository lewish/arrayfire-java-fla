package arrayfire;

public record TopKResult<T extends DataType<?, ?>, S extends Shape<?, ?, ?, ?>>(Tensor<T, S> values,
                                                                                Tensor<U32, S> indices) {
}
