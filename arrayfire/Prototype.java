package arrayfire;

public record Prototype<T extends DataType<?>, S extends Shape<?, ?, ?, ?>>(T type, S shape) {
}
