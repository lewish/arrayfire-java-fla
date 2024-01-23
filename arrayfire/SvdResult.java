package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.Num;
import arrayfire.numbers.U;

public record SvdResult<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>>(
        Tensor<T, D0, D0, U, U> u,
        Tensor<T, D0, U, U, U> s,
        Tensor<T, D1, D1, U, U> vt) {
}
