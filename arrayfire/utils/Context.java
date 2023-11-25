package arrayfire.utils;


import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Stack;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class Context {
  IdentityHashMap<Contextual<?>, Object> map;

  private static final ThreadLocal<Stack<Context>> context = ThreadLocal.withInitial(() -> {
    var stack = new Stack<Context>();
    stack.push(new Context());
    return stack;
  });

  public static Context current() {
    return context.get().peek();
  }

  private Context() {
    this.map = new IdentityHashMap<>();
  }

  public Map<Contextual<?>, Object> values() {
    return map;
  }

  public static boolean has(Contextual<?> contextual) {
    return current().map.containsKey(contextual);
  }

  @SuppressWarnings("unchecked")
  public static <T> T get(Contextual<T> contextual) {
    var map = current().map;
    return map.containsKey(contextual) ? (T) map.get(contextual) : contextual.defaultValue().orElseThrow();
  }

  public static AutoCloseable fork(Entry<?>... entries) {
    return fork(Arrays.stream(entries).collect(Collectors.toMap(Entry::contextual, Entry::value)));
  }

  public static AutoCloseable fork(Map<Contextual<?>, Object> map) {
    var newContext = new Context();
    newContext.map.putAll(current().map);
    newContext.map.putAll(map);

    context.get().push(newContext);
    return () -> {
      if (current() != newContext) {
        throw new RuntimeException("Can't close context, not the leaf context.");
      }
      context.get().pop();
      map.values().forEach(value -> {
        if (value instanceof Closeable c) {
          try {
            c.close();
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
        if (value instanceof AutoCloseable c) {
          try {
            c.close();
          } catch (Exception e) {
            if (e instanceof RuntimeException re) {
              throw re;
            }
            throw new RuntimeException(e);
          }
        }
      });
    };
  }


  public static void fork(Entry<?> e, Runnable fn) {
    fork(Stream.of(e).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static void fork(Entry<?> e, Entry<?> e2, Runnable fn) {
    fork(Stream.of(e, e2).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static void fork(Entry<?> e, Entry<?> e2, Entry<?> e3, Runnable fn) {
    fork(Stream.of(e, e2, e3).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static void fork(Map<Contextual<?>, Object> map, Runnable fn) {
    try (var ignored = fork(map)) {
      fn.run();
    } catch (Exception e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }

  public static <T> T fork(Entry<?> e, Supplier<T> fn) {
    return fork(Stream.of(e).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static <T> T fork(Entry<?> e, Entry<?> e2, Supplier<T> fn) {
    return fork(Stream.of(e, e2).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static <T> T fork(Entry<?> e, Entry<?> e2, Entry<?> e3, Supplier<T> fn) {
    return fork(Stream.of(e, e2, e3).collect(Collectors.toMap(Entry::contextual, Entry::value)), fn);
  }

  public static <T> T fork(Map<Contextual<?>, Object> map, Supplier<T> fn) {
    try (var ignored = fork(map)) {
      return fn.get();
    } catch (Exception e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }

  public static record Entry<T>(Contextual<T> contextual, T value) {
  }
}
