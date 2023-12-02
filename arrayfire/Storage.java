package arrayfire;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum Storage {
  DENSE(0), CSR(1), CSC(2), COO(3);

  private static final Map<Integer, Storage> codeMap = Arrays.stream(Storage.values())
      .collect(Collectors.toMap(Storage::code, Function.identity()));

  private final int code;

  Storage(int code) {
    this.code = code;
  }

  public static Storage fromCode(int code) {
    return codeMap.get(code);
  }

  public int code() {
    return code;
  }
}
