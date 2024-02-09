package arrayfire;

import arrayfire.utils.IdentityHashSet;

import java.util.*;
import java.util.stream.Collectors;

public class Scope {
    static final ThreadLocal<Scope> threadScope = ThreadLocal.withInitial(() -> null);
    private static final IdentityHashMap<MemoryContainer, Scope> containerScopes = new IdentityHashMap<>();
    private static final IdentityHashMap<Scope, Set<MemoryContainer>> scopeContainers = new IdentityHashMap<>();
    private final List<Operation> operations = new ArrayList<>();

    public static Scope current() {
        return threadScope.get();
    }

    public static void tidy(Runnable fn) {
        var previousScope = current();
        var scope = new Scope();
        try {
            threadScope.set(scope);
            fn.run();
        } finally {
            scope.dispose();
            threadScope.set(previousScope);
        }
    }

    /**
     * Permanently removes this memory container from the tracking system
     */
    public static void untrack(MemoryContainer mc) {
        scopeContainers.get(containerScopes.get(mc)).remove(mc);
        containerScopes.remove(mc);
    }

    public static Scope scopeOf(MemoryContainer memoryContainer) {
        return containerScopes.get(memoryContainer);
    }

    public static void move(MemoryContainer memoryContainer, Scope scope) {
        scopeContainers.get(containerScopes.get(memoryContainer)).remove(memoryContainer);
        containerScopes.put(memoryContainer, scope);
        scopeContainers.computeIfAbsent(scope, k -> IdentityHashSet.create()).add(memoryContainer);
    }

    public static Set<MemoryContainer> trackedContainers() {
        return containerScopes.keySet();
    }

    public static Set<Scope> trackedScopes() {
        return scopeContainers.keySet();
    }

    public static List<Array<?, ?>> trackedArrays() {
        return containerScopes
                   .keySet()
                   .stream()
                   .filter(mc -> mc instanceof Array)
                   .map(mc -> (Array<?, ?>) mc)
                   .collect(Collectors.toList());
    }

    public void dispose() {
        // Copy first to avoid concurrent modification exceptions.
        List.copyOf(scopeContainers.getOrDefault(this, Set.of())).forEach(mc -> {
            mc.dispose();
            containerScopes.remove(mc);
        });
        scopeContainers.remove(this);
    }

    public void register(MemoryContainer memoryContainer) {
        containerScopes.put(memoryContainer, this);
        scopeContainers.computeIfAbsent(this, k -> IdentityHashSet.create()).add(memoryContainer);
    }

    public void register(Operation operation) {
        // TODO: Only do this if eager execution is enabled.
        operation.outputs().forEach(this::register);
        operation.apply();
        operations.add(operation);
    }

    public List<Operation> operations() {
        return Collections.unmodifiableList(operations);
    }
}
