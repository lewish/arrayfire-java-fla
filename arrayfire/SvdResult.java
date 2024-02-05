package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

public record SvdResult<T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>>(Array<T, Shape<D0, D0, U, U>> u,
                                                                                       Array<T, Shape<D0, U, U, U>> s,
                                                                                       Array<T, Shape<D1, D1, U, U>> vt) {
}
