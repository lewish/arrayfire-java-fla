package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

public record SvdResult<T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>>(
    Tensor<T, R2<D0, D0>> u, Tensor<T, R1<D0>> s, Tensor<T, R2<D1, D1>> vt) {
}
