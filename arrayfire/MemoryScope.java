package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.utils.IdentityHashSet;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

public class MemoryScope {

    public static MemoryScope current() {
        return Scope.current().memory();
    }

    private static final IdentityHashMap<MemoryContainer, MemoryScope> containerScopes = new IdentityHashMap<>();
    private static final IdentityHashMap<MemoryScope, Set<MemoryContainer>> scopeContainers = new IdentityHashMap<>();

    private List<Operation> operations = new ArrayList<>();

    /**
     * Permanently removes this memory container from the tracking system
     */
    public static  void untrack(MemoryContainer mc) {
        scopeContainers.get(containerScopes.get(mc)).remove(mc);
        containerScopes.remove(mc);
    }

    public void dispose() {
        // Copy first to avoid concurrent modification exceptions.
        List.copyOf(scopeContainers.getOrDefault(this, Set.of())).forEach(MemoryContainer::dispose);
        scopeContainers.remove(this);
    }

    public static MemoryScope scopeOf(MemoryContainer memoryContainer) {
        return containerScopes.get(memoryContainer);
    }

    public void register(MemoryContainer memoryContainer) {
        containerScopes.put(memoryContainer, this);
        scopeContainers.computeIfAbsent(this, k -> IdentityHashSet.create()).add(memoryContainer);
    }

    public static void move(MemoryContainer memoryContainer, MemoryScope memoryScope) {
        scopeContainers.get(containerScopes.get(memoryContainer)).remove(memoryContainer);
        containerScopes.put(memoryContainer, memoryScope);
        scopeContainers.computeIfAbsent(memoryScope, k -> IdentityHashSet.create()).add(memoryContainer);
    }

    public static int trackedContainers() {
        return scopeContainers.keySet().size();
    }

    public void register(Operation operation) {
        // TODO: Only do this if eager execution is enabled.
        operation.outputs().forEach(this::register);
        operation.execute();
        operations.add(operation);
    }
}
