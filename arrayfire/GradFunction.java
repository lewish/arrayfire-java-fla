package arrayfire;

import arrayfire.numbers.Num;

import java.util.List;

@FunctionalInterface
interface GradFunction {

    List<Tensor<?, ?>> grads(Tensor<?, ?> resultGrads);

    interface Unary<RT extends Tensor<?, ?>, IT extends Tensor<?, ?>> {
        IT grads(RT result, RT grads);
    }

    interface Binary<RT extends Tensor<?, ?>, I0T extends Tensor<?, ?>, I1T extends Tensor<?, ?>> {
        TensorPair<I0T, I1T> grads(RT result, RT grads);
    }
}
