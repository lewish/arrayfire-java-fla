package arrayfire;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum Status {
    AF_SUCCESS(0), AF_ERR_NO_MEM(101), AF_ERR_DRIVER(102), AF_ERR_RUNTIME(103), AF_ERR_INVALID_ARRAY(201), AF_ERR_ARG(
            202), AF_ERR_SIZE(203), AF_ERR_TYPE(204), AF_ERR_DIFF_TYPE(205), AF_ERR_BATCH(207), AF_ERR_DEVICE(
            208), AF_ERR_NOT_SUPPORTED(301), AF_ERR_NOT_CONFIGURED(302), AF_ERR_NONFREE(303), AF_ERR_NO_DBL(
            401), AF_ERR_NO_GFX(402), AF_ERR_NO_HALF(403), AF_ERR_LOAD_LIB(501), AF_ERR_LOAD_SYM(
            502), AF_ERR_ARR_BKND_MISMATCH(503), AF_ERR_INTERNAL(998), AF_ERR_UNKNOWN(999),
    ;

    private static final Map<Integer, Status> codeMap = Arrays.stream(Status.values()).collect(
            Collectors.toMap(Status::code, Function.identity()));

    private final int code;

    Status(int code) {
        this.code = code;
    }

    public static Status fromCode(int code) {
        return codeMap.get(code);
    }

    public int code() {
        return code;
    }
}
