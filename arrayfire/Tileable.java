package arrayfire;

public record Tileable<T extends DataType<?>, S extends Shape<?, ?, ?, ?>>(Array<T, S> array) {
}
