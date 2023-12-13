package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

public record Tileable<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>>(
        Tensor<T, D0, D1, D2, D3> tensor) {
}
