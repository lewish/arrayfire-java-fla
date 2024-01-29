package arrayfire;

import arrayfire.DataType;
import arrayfire.Tensor;
import arrayfire.numbers.Num;

public record TensorTrio<I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>, I2T extends DataType<?, ?>, I2D0 extends Num<?>, I2D1 extends Num<?>, I2D2 extends Num<?>, I2D3 extends Num<?>>(
    Tensor<I0T, I0D0, I0D1, I0D2, I0D3> left, Tensor<I1T, I1D0, I1D1, I1D2, I1D3> middle,
    Tensor<I2T, I2D0, I2D1, I2D2, I2D3> right) {

}
