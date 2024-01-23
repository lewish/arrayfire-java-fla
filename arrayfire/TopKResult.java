package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.U32;
import arrayfire.numbers.Num;

public record TopKResult<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>>(
    Tensor<T, D0, D1, D2, D3> values, Tensor<U32, D0, D1, D2, D3> indices) {
}
