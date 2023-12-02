package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.U32;

public record TopKResult<T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number>(
    Tensor<T, D0, D1, D2, D3> values, Tensor<U32, D0, D1, D2, D3> indices) {
}
