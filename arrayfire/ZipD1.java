package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.U;

public record ZipD1<LT extends DataType<?, ?>, RT extends DataType<?, ?>, LD0 extends Number, RD0 extends Number, D1 extends Number>(
        Tensor<LT, LD0, D1, U, U> left, Tensor<RT, RD0, D1, U, U> right) {

}
