package arrayfire;

import arrayfire.autograd.GradFunction;
import arrayfire.numbers.Num;
import arrayfire.utils.IdentityHashSet;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Graph {

    private final Set<Params<?, ?, ?, ?, ?>> usedParams = IdentityHashSet.create();
    private final IdentityHashMap<Tensor<?, ?, ?, ?, ?>, Node> nodesByOutput = new IdentityHashMap<>();
    private final IdentityHashMap<Tensor<?, ?, ?, ?, ?>, List<Node>> dependentsByInput = new IdentityHashMap<>();

    public void add(Node node) {
        nodesByOutput.put(node.tensor, node);
        for (var input : node.inputs) {
            if (!dependentsByInput.containsKey(input)) {
                dependentsByInput.put(input, new ArrayList<>());
            }
            dependentsByInput.get(input).add(node);
        }
    }

    public Set<Tensor<?, ?, ?, ?, ?>> dependents(Tensor<?, ?, ?, ?, ?> tensor) {
        if (dependentsByInput.containsKey(tensor)) {
            return dependentsByInput.get(tensor).stream().map(node -> node.tensor).collect(
                    Collectors.toUnmodifiableSet());
        }
        return Collections.emptySet();
    }

    public Set<Tensor<?, ?, ?, ?, ?>> dependencies(Tensor<?, ?, ?, ?, ?> tensor) {
        if (nodesByOutput.containsKey(tensor)) {
            return nodesByOutput.get(tensor).inputs.stream().collect(Collectors.toUnmodifiableSet());
        }
        return Collections.emptySet();
    }

    public Collection<Tensor<?, ?, ?, ?, ?>> transitiveDependents(Tensor<?, ?, ?, ?, ?> tensor) {
        // This is an identity hash set.
        var queue = new ArrayDeque<Tensor<?, ?, ?, ?, ?>>(List.of(tensor));
        var set = Collections.newSetFromMap(new IdentityHashMap<Tensor<?, ?, ?, ?, ?>, Boolean>());
        while (!queue.isEmpty()) {
            var current = queue.poll();
            if (set.contains(current)) {
                continue;
            }
            set.add(current);
            queue.addAll(dependents(current));
        }
        return Collections.unmodifiableCollection(set);
    }

    public Set<Node> nodes() {
        return nodesByOutput.values().stream().collect(Collectors.toCollection(IdentityHashSet::create));
    }

    public <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> grads(
            Tensor<?, ?, ?, ?, ?> loss, Tensor<T, D0, D1, D2, D3> tensor) {
        return grads(loss, new Tensor[]{tensor}).get(tensor);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public void optimize(Tensor<?, ?, ?, ?, ?> loss) {
        var paramsToTensor = usedParams.stream().collect(Collectors.toMap(Function.identity(), Function.identity()));
        var grads = grads(loss, paramsToTensor.values().toArray(Tensor[]::new));
        for (var params : usedParams) {
            params.optimize((Tensor) grads.get(paramsToTensor.get(params)));
        }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public Grads grads(Tensor<?, ?, ?, ?, ?> loss, Tensor<?, ?, ?, ?, ?>... tensors) {
        var pruned = prune(loss, tensors);
        var queue = new ArrayDeque<>(pruned);
        var processedNodeOutputs = IdentityHashSet.<Tensor<?, ?, ?, ?, ?>>create();
        var gradsByOutput = new IdentityHashMap<Tensor<?, ?, ?, ?, ?>, Tensor<?, ?, ?, ?, ?>>();
        var parentScope = ArrayFire.scope();
        af.tidy(() -> {
            // We insert a sentinel value to ensure that we don't try to compute the gradient of the loss.
            gradsByOutput.put(loss, ArrayFire.constant(loss.type(), 1).tileAs((Tensor) loss));
            while (!queue.isEmpty()) {
                var current = queue.poll();
                // We can compute the gradient of this node if it was constructed from a gradable function.
                var node = nodesByOutput.get(current);
                if (node == null) {
                    // This is a source node, so we don't need to compute its gradient.
                    continue;
                }
                // If our dependents that are part of the pruned graph haven't all been computed yet, we can't compute this node's gradient yet.
                if (!processedNodeOutputs.containsAll(dependents(current).stream().filter(pruned::contains).toList())) {
                    queue.addLast(current);
                    continue;
                }
                if (node.gradFunction == null) {
                    throw new IllegalStateException(
                            String.format(
                                    "Attempting to compute the gradient of through a '%s' operation, but it does not support gradient propagation.",
                                    node.name));
                }
                var inputGrads = node.gradFunction.grads(gradsByOutput.get(current));
                for (var i = 0; i < node.inputs.size(); i++) {
                    var input = node.inputs.get(i);
                    var inputGrad = inputGrads.get(i);
                    if (!gradsByOutput.containsKey(input)) {
                        gradsByOutput.put(input, inputGrad);
                    } else {
                        gradsByOutput.put(input,
                                ArrayFire.add((Tensor) gradsByOutput.get(input), (Tensor) (inputGrad)));
                    }
                }
                processedNodeOutputs.add(current);
            }
            // Move all the gradients to the parent scope.
            Arrays.stream(tensors).map(gradsByOutput::get).forEach(
                    grad -> MemoryScope.move(grad, parentScope.memory()));
        });
        Grads grads = new Grads();
        for (var tensor : tensors) {
            grads.put(tensor, gradsByOutput.get(tensor));
        }
        return grads;
    }

    /**
     * Prunes the graph to only include the nodes that are required to compute the gradients of the given tensors back from the given loss.
     */
    public Set<Tensor<?, ?, ?, ?, ?>> prune(Tensor<?, ?, ?, ?, ?> loss, Tensor<?, ?, ?, ?, ?>... tensors) {
        var set = IdentityHashSet.<Tensor<?, ?, ?, ?, ?>>create();
        for (var tensor : tensors) {
            var transitiveDependents = transitiveDependents(tensor);
            if (!transitiveDependents.contains(loss)) {
                throw new IllegalStateException(
                        String.format("There is no path in the graph from %s to the given loss %s", tensor, loss));
            }
            set.addAll(transitiveDependents(tensor));
        }
        return set.stream().filter(dependent -> transitiveDependents(dependent).contains(loss)).collect(
                Collectors.toCollection(IdentityHashSet::create));
    }

    public void addParams(Params<?, ?, ?, ?, ?> params) {
        usedParams.add(params);
    }

    public record Node(String name, Tensor<?, ?, ?, ?, ?> tensor, List<? extends Tensor<?, ?, ?, ?, ?>> inputs,
                       GradFunction gradFunction) {
    }

    public static class Grads {
        private final Map<Tensor<?, ?, ?, ?, ?>, Tensor<?, ?, ?, ?, ?>> gradsByTensor = new IdentityHashMap<>();

        void put(Tensor<?, ?, ?, ?, ?> tensor, Tensor<?, ?, ?, ?, ?> grads) {
            gradsByTensor.put(tensor, grads);
        }

        @SuppressWarnings("unchecked")
        public <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> get(
                Tensor<T, D0, D1, D2, D3> tensor) {
            return (Tensor<T, D0, D1, D2, D3>) gradsByTensor.get(tensor);
        }
    }

}
