package arrayfire;

public record SortIndexResult<T extends DataType<?>, S extends Shape<?, ?, ?, ?>>(Tensor<T, S> values,
                                                                                  Tensor<U32, S> indices) {
}
