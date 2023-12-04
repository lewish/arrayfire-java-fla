package arrayfire;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum RandomEngineType {
    AF_RANDOM_ENGINE_PHILOX_4X32_10(100), AF_RANDOM_ENGINE_THREEFRY_2X32_16(200), AF_RANDOM_ENGINE_MERSENNE_GP11213(
            300),
    ;

    private static final Map<Integer, RandomEngineType> codeMap = Arrays.stream(RandomEngineType.values()).collect(
            Collectors.toMap(RandomEngineType::code, Function.identity()));

    private final int code;

    RandomEngineType(int code) {
        this.code = code;
    }

    public static RandomEngineType fromCode(int code) {
        return codeMap.get(code);
    }

    public int code() {
        return code;
    }
}
