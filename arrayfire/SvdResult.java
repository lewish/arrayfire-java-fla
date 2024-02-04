package arrayfire;

import arrayfire.numbers.Num;

public record SvdResult<T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>>(
    Array<T, R2<D0, D0>> u, Array<T, R1<D0>> s, Array<T, R2<D1, D1>> vt) {
}
