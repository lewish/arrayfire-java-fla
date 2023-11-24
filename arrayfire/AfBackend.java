package arrayfire;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum AfBackend {
  DEFAULT(0), CPU(1), CUDA(2), OPENCL(4),
  ;

  private static final Map<Integer, AfBackend> codeMap = Arrays.stream(AfBackend.values())
      .collect(Collectors.toMap(AfBackend::code, Function.identity()));

  private final int code;

  AfBackend(int code) {
    this.code = code;
  }

  public static AfBackend fromCode(int code) {
    return codeMap.get(code);
  }

  public static Set<AfBackend> fromBitmask(int mask) {
    var set = new HashSet<AfBackend>();
    for (AfBackend backend : AfBackend.values()) {
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
