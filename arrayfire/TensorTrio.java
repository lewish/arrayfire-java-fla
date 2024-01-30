package arrayfire;

public record TensorTrio<T1 extends Tensor<?, ?>, T2 extends Tensor<?, ?>, T3 extends Tensor<?, ?>>(T1 left, T2 middle,
                                                                                                    T3 right) {

}
