package arrayfire;

import arrayfire.datatypes.AfDataType;
import arrayfire.numbers.N;
import arrayfire.numbers.U;

import static arrayfire.ArrayFire.af;

public record Convolver2D<D0 extends Number, D1 extends Number, D2 extends Number, FD3 extends Number>(Shape<D0, D1, D2, U> input,
                                                                                                       Shape<?, ?, D2, FD3> filters,
                                                                                                       Shape<?, ?, U, U> stride,
                                                                                                       Shape<?, ?, U, U> padding,
                                                                                                       Shape<?, ?, U, U> dilation) {

  public Convolver2D(Shape<D0, D1, D2, U> input,
      Shape<?, ?, D2, FD3> filters,
      Shape<?, ?, U, U> stride,
      Shape<?, ?, U, U> padding) {
    this(input, filters, stride, padding, af.shape(1, 1));
  }

  public Convolver2D(Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters, Shape<?, ?, U, U> stride) {
    this(input, filters, stride, af.shape(0, 0));
  }

  public Convolver2D(Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters) {
    this(input, filters, af.shape(1, 1));
  }

  public <T extends AfDataType<?>,  D3 extends Number> Tensor<T, N, N, FD3, D3> convolve(
      Tensor<T, D0, D1, D2, D3> input,
      Tensor<T, ?, ?, D2, FD3> filters) {
    return af.convolve2(input, filters, stride, padding, dilation);
  }

  public Shape<N, N, FD3, U> outputShape() {
    // TODO: Doesn't factor in dilation.
    if (dilation.d0().intValue() != 1 || dilation.d1().intValue() != 1) {
      throw new UnsupportedOperationException("Dilation sizing not supported.");
    }
    return af.shape(
        af.n((input.d0().intValue() - filters.d0().intValue()) / stride.d0().intValue() + 1),
        af.n((input.d1().intValue() - filters.d1().intValue()) / stride.d1().intValue() + 1),
        filters.d3());
  }
}
