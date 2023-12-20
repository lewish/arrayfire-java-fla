package arrayfire;

import java.lang.foreign.Arena;
import java.util.HashSet;
import java.util.Set;

public class MemoryScope {

    private final Graph graph = new Graph();
    private final Set<MemoryContainer> memoryContainers = new HashSet<>();

    public Arena managedArena(MemoryContainer memoryContainer) {
        if (memoryContainers.contains(memoryContainer)) {
            throw new IllegalStateException("Memory container already tracked in this scope: " + memoryContainer);
        }
        memoryContainers.add(memoryContainer);
        return Arena.ofShared();
    }

    public Graph graph() {
        return graph;
    }

    public void track(MemoryContainer memoryContainer) {
        memoryContainers.add(memoryContainer);
    }

    public void untrack(MemoryContainer memoryContainer) {
        memoryContainers.remove(memoryContainer);
    }

    public void dispose() {
        memoryContainers.forEach(MemoryContainer::dispose);
        memoryContainers.clear();
    }
}
