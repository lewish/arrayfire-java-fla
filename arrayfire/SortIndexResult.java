package arrayfire;

public record SortIndexResult<T extends DataType<?>, S extends Shape<?, ?, ?, ?>>(Array<T, S> values,
                                                                                  Array<U32, S> indices) {
}
