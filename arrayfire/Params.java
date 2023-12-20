package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

public class Params<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> {
    private Tensor<T, D0, D1, D2, D3> tensor;

    public Params(Tensor<T, D0, D1, D2, D3> tensor) {
        this.tensor = tensor;
    }

    public void swap(Tensor<T, D0, D1, D2, D3> tensor) {
        this.tensor.release();
        this.tensor = tensor;
    }

    public void increment(Tensor<T, D0, D1, D2, D3> delta) {
        swap(af.add(tensor, delta));
    }

    public Tensor<T, D0, D1, D2, D3> get() {
        return tensor;
    }
}
