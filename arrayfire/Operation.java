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
    private final List<Array> inputs = new ArrayList<>();
    private final List<Array> outputs = new ArrayList<>();
    private Consumer<List<Array>> apply;
    private GradFunction grads;

    private boolean executed = false;

    public String name() {
        return name;
    }

    public List<Array> inputs() {
        return Collections.unmodifiableList(inputs);
    }

    public List<Array> outputs() {
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

        public <IT extends Array<?, ?>> Unary<IT> inputs(IT input) {
            operation.inputs.add(input);
            return new Unary<>();
        }

        public <I0T extends DataType<?>, I0S extends Shape<?, ?, ?, ?>, I1T extends DataType<?>, I1S extends Shape<?, ?, ?, ?>> Binary<Array<I0T, I0S>, Array<I1T, I1S>> inputs(
            Array<I0T, I0S> left, Array<I1T, I1S> right) {
            operation.inputs.add(left);
            operation.inputs.add(right);
            return new Binary<>();
        }

        public class Nullary {

            public <OT extends DataType<?>, OS extends Shape<?, ?, ?, ?>> Single<Array<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Array<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends Array<?, ?>> {

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

        public class Unary<IT extends Array<?, ?>> {

            public None outputs() {
                return new None();
            }

            public <OT extends DataType<?>, OS extends Shape<?, ?, ?, ?>> Single<Array<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Array<>(prototype));
                return new Single<>();
            }

            public <O0T extends DataType<?>, O0S extends Shape<?, ?, ?, ?>, O1T extends DataType<?>, O1S extends Shape<?, ?, ?, ?>> Pair<Array<O0T, O0S>, Array<O1T, O1S>> outputs(
                Prototype<O0T, O0S> left, Prototype<O1T, O1S> right) {
                operation.outputs.add(new Array<>(left));
                operation.outputs.add(new Array<>(right));
                return new Pair<>();
            }

            public <O0T extends DataType<?>, O0S extends Shape<?, ?, ?, ?>, O1T extends DataType<?>, O1S extends Shape<?, ?, ?, ?>, O2T extends DataType<?>, O2S extends Shape<?, ?, ?, ?>> Trio<Array<O0T, O0S>, Array<O1T, O1S>, Array<O2T, O2S>> outputs(
                Prototype<O0T, O0S> left, Prototype<O1T, O1S> middle, Prototype<O2T, O2S> right) {
                operation.outputs.add(new Array<>(left));
                operation.outputs.add(new Array<>(middle));
                operation.outputs.add(new Array<>(right));
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

            public class Single<OT extends Array<?, ?>> {

                public Single<OT> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT> grads(GradFunction.Unary<OT, IT> unaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = unaryGradFunction.grads((OT) operation.outputs.getFirst(), (OT) grads.getFirst());
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

            public class Pair<O0T extends Array<?, ?>, O1T extends Array<?, ?>> {

                public Pair<O0T, O1T> operation(Functions.Function2<MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Pair<O0T, O1T> grads(GradFunction.UnaryPair<O0T, O1T, IT> unaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = unaryGradFunction.grads(new ArrayPair<>((O0T) operation.outputs.getFirst(), (O1T) operation.outputs.get(1)), new ArrayPair<>((O0T) grads.getFirst(), (O1T) grads.get(1)));
                        return List.of(inputGrad);
                    };
                    return this;
                }

                @SuppressWarnings("unchecked")
                public ArrayPair<O0T, O1T> build() {
                    af.scope().register(operation);
                    return new ArrayPair<>((O0T) operation.outputs.getFirst(), (O1T) operation.outputs.get(1));
                }
            }

            public class Trio<O0T extends Array<?, ?>, O1T extends Array<?, ?>, O2T extends Array<?, ?>> {

                public Trio<O0T, O1T, O2T> operation(
                    Functions.Function3<MemorySegment, MemorySegment, MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(
                        () -> function.apply(outputs.getFirst().segment(), outputs.get(1).segment(),
                            outputs.get(2).segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public ArrayTrio<O0T, O1T, O2T> build() {
                    af.scope().register(operation);
                    return new ArrayTrio<>((O0T) operation.outputs.getFirst(), (O1T) operation.outputs.get(1),
                        (O2T) operation.outputs.get(2));
                }
            }
        }

        public class Binary<I0T extends Array<?, ?>, I1T extends Array<?, ?>> {

            public <OT extends DataType<?>, OS extends Shape<?, ?, ?, ?>> Single<Array<OT, OS>> outputs(
                Prototype<OT, OS> prototype) {
                operation.outputs.add(new Array<>(prototype));
                return new Single<>();
            }

            public class Single<OT extends Array<?, ?>> {

                public Single<OT> operation(Function<MemorySegment, Integer> function) {
                    operation.apply = (outputs) -> af.handleStatus(() -> function.apply(outputs.getFirst().segment()));
                    return this;
                }

                @SuppressWarnings("unchecked")
                public Single<OT> grads(GradFunction.Binary<OT, I0T, I1T> binaryGradFunction) {
                    operation.grads = (grads) -> {
                        var inputGrad = binaryGradFunction.grads((OT) operation.outputs.getFirst(), (OT) grads.getFirst());
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