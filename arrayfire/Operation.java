package arrayfire;

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

    GradFunction grads() {
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

        public <IT extends Tensor<?, ?>> Unary<IT> inputs(IT input) {
            operation.inputs.add(input);
            return new Unary<>();
        }

        public <I0T extends DataType<?, ?>, I0S extends Shape<?, ?, ?, ?>, I1T extends DataType<?, ?>, I1S extends Shape<?, ?, ?, ?>> Binary<Tensor<I0T, I0S>, Tensor<I1T, I1S>> inputs(
            Tensor<I0T, I0S> left, Tensor<I1T, I1S> right) {
            operation.inputs.add(left);
            operation.inputs.add(right);
            return new Binary<>();
        }

        public class Nullary {

            public <OT extends DataType<?, ?>, OS extends Shape<?, ?, ?, ?>> Single<Tensor<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends Tensor<?, ?>> {

                public Single<OT> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public OT build() {
                    af.scope().register(operation);
                    return (OT) operation.outputs.getFirst();
                }
            }
        }

        public class Unary<IT extends Tensor<?, ?>> {

            public None outputs() {
                return new None();
            }

            public <OT extends DataType<?, ?>, OS extends Shape<?, ?, ?, ?>> Single<Tensor<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public <O0T extends DataType<?, ?>, O0S extends Shape<?, ?, ?, ?>, O1T extends DataType<?, ?>, O1S extends Shape<?, ?, ?, ?>> Pair<Tensor<O0T, O0S>, Tensor<O1T, O1S>> outputs(
                Prototype<O0T, O0S> left, Prototype<O1T, O1S> right) {
                operation.outputs.add(new Tensor<>(left));
                operation.outputs.add(new Tensor<>(right));
                return new Pair<>();
            }

            public <O0T extends DataType<?, ?>, O0S extends Shape<?, ?, ?, ?>, O1T extends DataType<?, ?>, O1S extends Shape<?, ?, ?, ?>, O2T extends DataType<?, ?>, O2S extends Shape<?, ?, ?, ?>> Trio<Tensor<O0T, O0S>, Tensor<O1T, O1S>, Tensor<O2T, O2S>> outputs(
                Prototype<O0T, O0S> left, Prototype<O1T, O1S> middle, Prototype<O2T, O2S> right) {
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
                    af.scope().register(operation);
                    return operation;
                }
            }

            public class Single<OT extends Tensor<?, ?>> {

                public Single<OT> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT> grads(GradFunction.Unary<OT, IT> unaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = unaryGradFunction.grads((OT) operation.outputs.getFirst(), (OT) grads);
                        return List.of(inputGrad);
                    };
                    return this;
                }

                @SuppressWarnings("unchecked")
                public OT build() {
                    af.scope().register(operation);
                    return (OT) operation.outputs.getFirst();
                }
            }

            public class Pair<O0T extends Tensor<?, ?>, O1T extends Tensor<?, ?>> {

                public Pair<O0T, O1T> operation(Functions.Function2<MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public TensorPair<O0T, O1T> build() {
                    af.scope().register(operation);
                    return new TensorPair<>((O0T) operation.outputs.getFirst(), (O1T) operation.outputs.get(1));
                }
            }

            public class Trio<O0T extends Tensor<?, ?>, O1T extends Tensor<?, ?>, O2T extends Tensor<?, ?>> {

                public Trio<O0T, O1T, O2T> operation(
                    Functions.Function3<MemorySegment, MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment(),
                            outputs.get(2).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public TensorTrio<O0T, O1T, O2T> build() {
                    af.scope().register(operation);
                    return new TensorTrio<>((O0T) operation.outputs.getFirst(), (O1T) operation.outputs.get(1),
                        (O2T) operation.outputs.get(2));
                }
            }
        }

        public class Binary<I0T extends Tensor<?, ?>, I1T extends Tensor<?, ?>> {

            public <OT extends DataType<?, ?>, OS extends Shape<?, ?, ?, ?>> Single<Tensor<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Tensor<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends Tensor<?, ?>> {

                public Single<OT> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT> grads(GradFunction.Binary<OT, I0T, I1T> binaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = binaryGradFunction.grads((OT) operation.outputs.getFirst(), (OT) grads);
                        return List.of(inputGrad.left(), inputGrad.right());
                    };
                    return this;
                }

                @SuppressWarnings("unchecked")
                public OT build() {
                    af.scope().register(operation);
                    return (OT) operation.outputs.getFirst();
                }
            }
        }
    }
}