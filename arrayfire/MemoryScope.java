package arrayfire;

import arrayfire.utils.IdentityHashSet;

import java.util.IdentityHashMap;
import java.util.Set;

public class MemoryScope {

    public static MemoryScope current() {
        return Scope.current().memory();
    }

    private static final IdentityHashMap<MemoryContainer, MemoryScope> containerScopes = new IdentityHashMap<>();
    private static final IdentityHashMap<MemoryScope, Set<MemoryContainer>> scopeContainers = new IdentityHashMap<>();

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
