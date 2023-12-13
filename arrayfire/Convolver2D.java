package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;
import arrayfire.numbers.N;
import arrayfire.numbers.U;


public record Convolver2D<D0 extends IntNumber<D0>, D1 extends IntNumber<D1>, D2 extends IntNumber<D2>, FD3 extends IntNumber<FD3>>(
        Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters, Shape<?, ?, U, U> stride, Shape<?, ?, U, U> padding,
        Shape<?, ?, U, U> dilation) {

    public Convolver2D(Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters, Shape<?, ?, U, U> stride,
                       Shape<?, ?, U, U> padding) {
        this(input, filters, stride, padding, af.shape(1, 1));
    }

    public Convolver2D(Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters, Shape<?, ?, U, U> stride) {
        this(input, filters, stride, af.shape(0, 0));
    }

    public Convolver2D(Shape<D0, D1, D2, U> input, Shape<?, ?, D2, FD3> filters) {
        this(input, filters, af.shape(1, 1));
    }

    public <T extends DataType<?, ?>, D3 extends IntNumber<?>> Tensor<T, N, N, FD3, D3> convolve(
            Tensor<T, D0, D1, D2, D3> input, Tensor<T, ?, ?, D2, FD3> filters) {
        return af.convolve2(input, filters, stride, padding, dilation);
    }

    public Shape<N, N, FD3, U> outputShape() {
        // TODO: Doesn't factor in dilation.
        if (dilation.d0().size() != 1 || dilation.d1().size() != 1) {
            throw new UnsupportedOperationException("Dilation sizing not supported.");
        }
        return af.shape(af.n((input.d0().size() - filters.d0().size()) / stride.d0().size() + 1),
                af.n((input.d1().size() - filters.d1().size()) / stride.d1().size() + 1), filters.d3());
    }
}
