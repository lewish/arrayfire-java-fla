package arrayfire;

import java.util.List;

@FunctionalInterface
interface GradFunction {

    List<Array<?, ?>> grads(Array<?, ?> resultGrads);

    interface Unary<RT extends Array<?, ?>, IT extends Array<?, ?>> {
        IT grads(RT result, RT grads);
    }

    interface Binary<RT extends Array<?, ?>, I0T extends Array<?, ?>, I1T extends Array<?, ?>> {
        ArrayPair<I0T, I1T> grads(RT result, RT grads);
    }
}
