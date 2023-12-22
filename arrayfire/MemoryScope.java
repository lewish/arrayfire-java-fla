package arrayfire;

import arrayfire.utils.IdentityHashSet;

import java.util.IdentityHashMap;
import java.util.Set;

public class MemoryScope {

    private static final ThreadLocal<MemoryScope> threadScope = ThreadLocal.withInitial(() -> null);
    private static final IdentityHashMap<MemoryContainer, MemoryScope> containerScopes = new IdentityHashMap<>();
    private static final IdentityHashMap<MemoryScope, Set<MemoryContainer>> scopeContainers = new IdentityHashMap<>();

    private final Graph graph = new Graph();

    public static MemoryScope current() {
        return threadScope.get();
    }

    public Graph graph() {
        return graph;
    }

    public static void tidy(Runnable fn) {
        var previousScope = current();
        try {
            threadScope.set(new MemoryScope());
            fn.run();
        } finally {
            threadScope.get().dispose();
            threadScope.set(previousScope);
        }
    }

    public void dispose() {
        scopeContainers.getOrDefault(this, Set.of()).forEach(mc -> {
            mc.dispose();
            containerScopes.remove(mc);
        });
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
}
