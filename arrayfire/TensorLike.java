package arrayfire;

import arrayfire.datatypes.AfDataType;

public interface TensorLike<T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> {
  Tensor<T, D0, D1, D2, D3> tensor();

  Shape<D0, D1, D2, D3> shape();
}
