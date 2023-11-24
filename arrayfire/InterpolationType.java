package arrayfire;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum InterpolationType {
  NEAREST(0), LINEAR(1), BILINEAR(2), CUBIC(3), LOWER(4), LINEAR_COSINE(5), BILINEAR_COSINE(6), BICUBIC(7), CUBIC_SPLINE(
      8), BICUBIC_SPLINE(9),
  ;

  private static final Map<Integer, InterpolationType> codeMap = Arrays.stream(InterpolationType.values())
      .collect(Collectors.toMap(InterpolationType::code, Function.identity()));

  private final int code;

  InterpolationType(int code) {
    this.code = code;
  }

  public static InterpolationType fromCode(int code) {
    return codeMap.get(code);
  }

  public int code() {
    return code;
  }
}
