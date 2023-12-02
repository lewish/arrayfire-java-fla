package arrayfire;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum Backend {
  DEFAULT(0), CPU(1), CUDA(2), OPENCL(4),
  ;

  private static final Map<Integer, Backend> codeMap = Arrays.stream(Backend.values())
      .collect(Collectors.toMap(Backend::code, Function.identity()));

  private final int code;

  Backend(int code) {
    this.code = code;
  }

  public static Backend fromCode(int code) {
    return codeMap.get(code);
  }

  public static Set<Backend> fromBitmask(int mask) {
    var set = new HashSet<Backend>();
    for (Backend backend : Backend.values()) {
      if ((mask & backend.code()) > 0) {
        set.add(backend);
      }
    }
    return Collections.unmodifiableSet(set);
  }

  public int code() {
    return code;
  }
}
