package arrayfire;

public class Scope {

    private static final ThreadLocal<Scope> threadScope = ThreadLocal.withInitial(() -> null);

    private final MemoryScope memory = new MemoryScope();
    private final Graph graph = new Graph();

    public static Scope current() {
        return threadScope.get();
    }

    public Graph graph() {
        return graph;
    }

    public MemoryScope memory() {
        return memory;
    }

    public static void tidy(Runnable fn) {
        var previousScope = current();
        var scope = new Scope();
        try {
            threadScope.set(scope);
            fn.run();
        } finally {
            scope.memory.dispose();
            threadScope.set(previousScope);
        }
    }
}
