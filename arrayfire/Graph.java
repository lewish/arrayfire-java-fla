package arrayfire;

import arrayfire.utils.IdentityHashSet;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

@SuppressWarnings("rawtypes")
public class Graph {

    private final IdentityHashMap<Array, Operation> nodesByOutput = new IdentityHashMap<>();
    private final IdentityHashMap<Array, List<Operation>> dependentsByInput = new IdentityHashMap<>();
    private final Set<Params> inputParams = IdentityHashSet.create();

    public Graph(List<Operation> operations) {
        operations.forEach(operation -> {
            operation.outputs().forEach(tensor -> nodesByOutput.put(tensor, operation));
            for (var input : operation.inputs()) {
                if (!dependentsByInput.containsKey(input)) {
                    dependentsByInput.put(input, new ArrayList<>());
                }
                dependentsByInput.get(input).add(operation);
            }
        });
        operations
            .stream()
            .flatMap(node -> node.inputs().stream())
            .filter(input -> input instanceof Params)
            .map(input -> (Params) input)
            .forEach(inputParams::add);

    }

    public Set<Array> dependents(Array array) {
        if (dependentsByInput.containsKey(array)) {
            return dependentsByInput
                       .get(array)
                       .stream()
                       .flatMap(node -> node.outputs().stream())
                       .collect(Collectors.toUnmodifiableSet());
        }
        return Collections.emptySet();
    }

    public Set<Array> dependencies(Array array) {
        if (nodesByOutput.containsKey(array)) {
            return nodesByOutput.get(array).inputs().stream().collect(Collectors.toUnmodifiableSet());
        }
        return Collections.emptySet();
    }

    public Collection<Array> transitiveDependents(Array array) {
        // This is an identity hash set.
        var queue = new ArrayDeque<>(List.of(array));
        var set = Collections.newSetFromMap(new IdentityHashMap<Array, Boolean>());
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

    @SuppressWarnings({"rawtypes", "unchecked"})
    public void optimize(Array loss) {
        var paramsToTensor = inputParams.stream().collect(Collectors.toMap(Function.identity(), Function.identity()));
        var grads = grads(loss, paramsToTensor.values().toArray(Array[]::new));
        for (var params : inputParams) {
            params.optimize(grads.get(paramsToTensor.get(params)));
        }
    }

    public <T extends Array<?, ?>> T grads(Array loss, T tensor) {
        var grads = grads(loss, new Array[]{tensor});
        return grads.get(tensor);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public Grads grads(Array loss, Array... arrays) {
        var pruned = prune(loss, arrays);
        var queue = new ArrayDeque<>(pruned);
        var processedNodeOutputs = IdentityHashSet.<Array>create();
        var gradsByOutput = new IdentityHashMap<Array, Array>();

        // We insert a sentinel value to ensure that we don't try to compute the gradient of the loss.
        gradsByOutput.put(loss, ArrayFire.constant(loss.type(), 1).tileAs((Array) loss));
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
            if (node.grads() == null) {
                throw new IllegalStateException(String.format(
                    "Attempting to compute the gradient of through a '%s' operation, but it does not support gradient propagation.",
                    node.name()));
            }
            var outputGrads = node.outputs().stream().map(gradsByOutput::get).toList();
            var inputGrads = node.grads().grads(outputGrads);
            for (var i = 0; i < node.inputs().size(); i++) {
                var input = node.inputs().get(i);
                var inputGrad = inputGrads.get(i);
                if (!gradsByOutput.containsKey(input)) {
                    gradsByOutput.put(input, inputGrad);
                } else {
                    gradsByOutput.put(input, ArrayFire.add((Array) gradsByOutput.get(input), (Array) (inputGrad)));
                }
            }
            processedNodeOutputs.add(current);
        }

        Grads grads = new Grads();
        for (var tensor : arrays) {
            grads.put(tensor, gradsByOutput.get(tensor));
        }
        return grads;
    }

    /**
     * Prunes the graph to only include the nodes that are required to compute the gradients of the given arrays back from the given loss.
     */
    public Set<Array> prune(Array loss, Array... arrays) {
        var set = IdentityHashSet.<Array>create();
        for (var tensor : arrays) {
            var transitiveDependents = transitiveDependents(tensor);
            if (!transitiveDependents.contains(loss)) {
                throw new IllegalStateException(
                    String.format("There is no path in the graph from %s to the given loss %s", tensor, loss));
            }
            set.addAll(transitiveDependents(tensor));
        }
        return set
                   .stream()
                   .filter(dependent -> transitiveDependents(dependent).contains(loss))
                   .collect(Collectors.toCollection(IdentityHashSet::create));
    }

    public static class Grads {
        private final Map<Array, Array> gradsByTensor = new IdentityHashMap<>();

        void put(Array array, Array grads) {
            gradsByTensor.put(array, grads);
        }

        @SuppressWarnings("unchecked")
        public <T extends Array<?, ?>> T get(T tensor) {
            return (T) gradsByTensor.get(tensor);
        }
    }

}
