package arrayfire;

import java.util.List;

@FunctionalInterface
interface GradFunction {

    @SuppressWarnings("rawtypes")
    List<Array<?, ?>> grads(List<Array> resultGrads);

    interface Unary<RT extends Array<?, ?>, IT extends Array<?, ?>> {
        IT grads(RT result, RT grads);
    }

    interface UnaryPair<RT1 extends Array<?, ?>, RT2 extends Array<?, ?>, IT extends Array<?, ?>> {
        IT grads(ArrayPair<RT1, RT2> results, ArrayPair<RT1, RT2> grads);
    }

    interface Binary<RT extends Array<?, ?>, I0T extends Array<?, ?>, I1T extends Array<?, ?>> {
        ArrayPair<I0T, I1T> grads(RT result, RT grads);
    }
}
