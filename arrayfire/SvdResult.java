package arrayfire;

import arrayfire.datatypes.AfDataType;
import arrayfire.numbers.U;

public record SvdResult<T extends AfDataType<?, ?>, D0 extends Number, D1 extends Number>(Tensor<T, D1, D1, U, U> u,
                                                                                       Tensor<T, D1, U, U, U> s,
                                                                                       Tensor<T, D0, D0, U, U> v) {
}
