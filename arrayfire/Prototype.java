package arrayfire;

import arrayfire.numbers.Num;

public record Prototype<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>>(
    T type, Shape<D0, D1, D2, D3> shape) {
}
