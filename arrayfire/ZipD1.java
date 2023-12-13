package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;
import arrayfire.numbers.U;

public record ZipD1<LT extends DataType<?, ?>, RT extends DataType<?, ?>, LD0 extends IntNumber<LD0>, RD0 extends IntNumber<RD0>, D1 extends IntNumber<D1>>(
        Tensor<LT, LD0, D1, U, U> left, Tensor<RT, RD0, D1, U, U> right) {

}
