package arrayfire;

public record TopKResult<T extends DataType<?>, S extends Shape<?, ?, ?, ?>>(Array<T, S> values,
                                                                             Array<U32, S> indices) {
}
