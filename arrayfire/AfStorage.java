package arrayfire;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum AfStorage {
  DENSE(0), CSR(1), CSC(2), COO(3);

  private static final Map<Integer, AfStorage> codeMap = Arrays.stream(AfStorage.values())
      .collect(Collectors.toMap(AfStorage::code, Function.identity()));

  private final int code;

  AfStorage(int code) {
    this.code = code;
  }

  public static AfStorage fromCode(int code) {
    return codeMap.get(code);
  }

  public int code() {
    return code;
  }
}
