package arrayfire.autograd;

import arrayfire.ArrayFire;
import arrayfire.Tensor;
import arrayfire.af;
import arrayfire.utils.IdentityHashSet;

import java.util.*;
import java.util.stream.Collectors;

public class Graph {

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

    @SuppressWarnings("unchecked")
    public <T extends Tensor<?, ?, ?, ?, ?>> T grads(Tensor<?, ?, ?, ?, ?> loss, T tensor) {
        return (T) grads(loss, new Tensor[]{tensor}).getFirst();
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public List<Tensor<?, ?, ?, ?, ?>> grads(Tensor<?, ?, ?, ?, ?> loss, Tensor<?, ?, ?, ?, ?>... tensors) {
        var pruned = prune(loss, tensors);
        var queue = new ArrayDeque<>(pruned);
        var processedNodeOutputs = IdentityHashSet.<Tensor<?, ?, ?, ?, ?>>create();
        var gradsByOutput = new IdentityHashMap<Tensor<?, ?, ?, ?, ?>, Tensor<?, ?, ?, ?, ?>>();
        var parentScope = ArrayFire.memoryScope();
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
                    grad -> ArrayFire.moveScope(grad, ArrayFire.memoryScope(), parentScope));
        });
        return Arrays.stream(tensors).map(gradsByOutput::get).collect(Collectors.toUnmodifiableList());
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

    public record Node(String name, Tensor<?, ?, ?, ?, ?> tensor, List<Tensor<?, ?, ?, ?, ?>> inputs,
                       GradFunction gradFunction) {
    }

}
