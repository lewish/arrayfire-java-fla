package arrayfire.datatypes;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum AfDataTypeEnum {
  F32(0), C32(1), F64(2), C64(3), B8(4), S32(5), U32(6), U8(7), S64(8), U64(9), S16(10), U16(11), F16(12),
  ;

  private static final Map<Integer, AfDataTypeEnum> codeMap = Arrays.stream(AfDataTypeEnum.values())
      .collect(Collectors.toMap(AfDataTypeEnum::code, Function.identity()));

  private final int code;

  AfDataTypeEnum(int code) {
    this.code = code;
  }

  public static AfDataTypeEnum fromCode(int code) {
    return codeMap.get(code);
  }

  public int code() {
    return code;
  }
}
