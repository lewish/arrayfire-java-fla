package arrayfire;

import arrayfire.autograd.GradFunction;
import arrayfire.numbers.Num;
import arrayfire.utils.Functions;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

@SuppressWarnings("rawtypes")
public class Operation {
    private String name;
    private final List<Tensor> inputs = new ArrayList<>();
    private final List<Tensor> outputs = new ArrayList<>();
    private Consumer<List<Tensor>> apply;
    private GradFunction grads;

    private boolean executed = false;

    public String name() {
        return name;
    }

    public List<Tensor> inputs() {
        return Collections.unmodifiableList(inputs);
    }

    public List<Tensor> outputs() {
        return Collections.unmodifiableList(outputs);
    }

    public void apply() {
        if (!executed) {
            apply.accept(outputs);
            executed = true;
        }
    }

    public GradFunction grads() {
        return grads;
    }

    public static class Builder {

        private final Operation operation = new Operation();

        public Builder name(String name) {
            operation.name = name;
            return this;
        }

        public Nullary inputs() {
            return new Nullary();
        }

        public <I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>> Unary<I0T, I0D0, I0D1, I0D2, I0D3> inputs(
            Tensor<I0T, I0D0, I0D1, I0D2, I0D3> input) {
            operation.inputs.add(input);
            return new Unary<>();
        }

        public <I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>> Binary<I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> inputs(
            Tensor<I0T, I0D0, I0D1, I0D2, I0D3> left, Tensor<I1T, I1D0, I1D1, I1D2, I1D3> right) {
            operation.inputs.add(left);
            operation.inputs.add(right);
            return new Binary<>();
        }

        public class Nullary {

            public <OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Single<OT, OD0, OD1, OD2, OD3> outputs(
                Prototype<OT, OD0, OD1, OD2, OD3> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> {

                public Single<OT, OD0, OD1, OD2, OD3> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Tensor<OT, OD0, OD1, OD2, OD3> build() {
                    af.scope().memory().register(operation);
                    return (Tensor<OT, OD0, OD1, OD2, OD3>) operation.outputs.getFirst();
                }
            }
        }

        public class Unary<I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>> {

            public <OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> None outputs() {
                return new None();
            }

            public <OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Single<OT, OD0, OD1, OD2, OD3> outputs(
                Prototype<OT, OD0, OD1, OD2, OD3> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public <O0T extends DataType<?, ?>, O0D0 extends Num<?>, O0D1 extends Num<?>, O0D2 extends Num<?>, O0D3 extends Num<?>, O1T extends DataType<?, ?>, O1D0 extends Num<?>, O1D1 extends Num<?>, O1D2 extends Num<?>, O1D3 extends Num<?>> Pair<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3> outputs(
                Prototype<O0T, O0D0, O0D1, O0D2, O0D3> left, Prototype<O1T, O1D0, O1D1, O1D2, O1D3> right) {
                operation.outputs.add(new Tensor<>(left));
                operation.outputs.add(new Tensor<>(right));
                return new Pair<>();
            }

            public <O0T extends DataType<?, ?>, O0D0 extends Num<?>, O0D1 extends Num<?>, O0D2 extends Num<?>, O0D3 extends Num<?>, O1T extends DataType<?, ?>, O1D0 extends Num<?>, O1D1 extends Num<?>, O1D2 extends Num<?>, O1D3 extends Num<?>, O2T extends DataType<?, ?>, O2D0 extends Num<?>, O2D1 extends Num<?>, O2D2 extends Num<?>, O2D3 extends Num<?>> Trio<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3, O2T, O2D0, O2D1, O2D2, O2D3> outputs(
                Prototype<O0T, O0D0, O0D1, O0D2, O0D3> left, Prototype<O1T, O1D0, O1D1, O1D2, O1D3> middle,
                Prototype<O2T, O2D0, O2D1, O2D2, O2D3> right) {
                operation.outputs.add(new Tensor<>(left));
                operation.outputs.add(new Tensor<>(middle));
                operation.outputs.add(new Tensor<>(right));
                return new Trio<>();
            }

            public class None {

                public None operation(Runnable function) {
                    operation.apply = (outputs) -> function.run();
                    return this;
                }

                public Operation build() {
                    af.scope().memory().register(operation);
                    return operation;
                }
            }

            public class Single<OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> {

                public Single<OT, OD0, OD1, OD2, OD3> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT, OD0, OD1, OD2, OD3> grads(
                    GradFunction.Unary<OT, OD0, OD1, OD2, OD3, I0T, I0D0, I0D1, I0D2, I0D3> unaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = unaryGradFunction.grads(operation.outputs.getFirst(), (Tensor) grads);
                        return List.of(inputGrad);
                    };
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Tensor<OT, OD0, OD1, OD2, OD3> build() {
                    af.scope().memory().register(operation);
                    return (Tensor<OT, OD0, OD1, OD2, OD3>) operation.outputs.getFirst();
                }
            }

            public class Pair<O0T extends DataType<?, ?>, O0D0 extends Num<?>, O0D1 extends Num<?>, O0D2 extends Num<?>, O0D3 extends Num<?>, O1T extends DataType<?, ?>, O1D0 extends Num<?>, O1D1 extends Num<?>, O1D2 extends Num<?>, O1D3 extends Num<?>> {

                public Pair<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3> operation(
                    Functions.Function2<MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public GradFunction.TensorPair<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3> build() {
                    af.scope().memory().register(operation);
                    return new GradFunction.TensorPair<>(
                        (Tensor<O0T, O0D0, O0D1, O0D2, O0D3>) operation.outputs.getFirst(),
                        (Tensor<O1T, O1D0, O1D1, O1D2, O1D3>) operation.outputs.get(1));
                }
            }

            public class Trio<O0T extends DataType<?, ?>, O0D0 extends Num<?>, O0D1 extends Num<?>, O0D2 extends Num<?>, O0D3 extends Num<?>, O1T extends DataType<?, ?>, O1D0 extends Num<?>, O1D1 extends Num<?>, O1D2 extends Num<?>, O1D3 extends Num<?>, O2T extends DataType<?, ?>, O2D0 extends Num<?>, O2D1 extends Num<?>, O2D2 extends Num<?>, O2D3 extends Num<?>> {

                public Trio<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3, O2T, O2D0, O2D1, O2D2, O2D3> operation(
                    Functions.Function3<MemorySegment, MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment(),
                            outputs.get(2).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public GradFunction.TensorTrio<O0T, O0D0, O0D1, O0D2, O0D3, O1T, O1D0, O1D1, O1D2, O1D3, O2T, O2D0, O2D1, O2D2, O2D3> build() {
                    af.scope().memory().register(operation);
                    return new GradFunction.TensorTrio<>(
                        (Tensor<O0T, O0D0, O0D1, O0D2, O0D3>) operation.outputs.getFirst(),
                        (Tensor<O1T, O1D0, O1D1, O1D2, O1D3>) operation.outputs.get(1),
                        (Tensor<O2T, O2D0, O2D1, O2D2, O2D3>) operation.outputs.get(2));
                }
            }
        }

        public class Binary<I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>> {

            public <OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Single<OT, OD0, OD1, OD2, OD3> outputs(
                Prototype<OT, OD0, OD1, OD2, OD3> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> {

                public Single<OT, OD0, OD1, OD2, OD3> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT, OD0, OD1, OD2, OD3> grads(
                    GradFunction.Binary<OT, OD0, OD1, OD2, OD3, I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> binaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = binaryGradFunction.grads(operation.outputs.getFirst(), (Tensor) grads);
                        return List.of(inputGrad.left(), inputGrad.right());
                    };
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Tensor<OT, OD0, OD1, OD2, OD3> build() {
                    af.scope().memory().register(operation);
                    return (Tensor<OT, OD0, OD1, OD2, OD3>) operation.outputs.getFirst();
                }
            }
        }
    }
}