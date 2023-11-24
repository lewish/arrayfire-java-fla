package arrayfire;

import arrayfire.datatypes.AfDataType;
import arrayfire.numbers.U;

public record ZipD1<LT extends AfDataType<?>, RT extends AfDataType<?>, LD0 extends Number, RD0 extends Number, D1 extends Number>(
    Tensor<LT, LD0, D1, U, U> left, Tensor<RT, RD0, D1, U, U> right) {

}
