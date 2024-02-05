package(default_visibility = ["//visibility:public"])

genrule(
    name = "generate_capi",
    srcs = [
        "@arrayfire//:include/arrayfire.h",
        "@arrayfire//:include_files",
        "@jextract//:bin/jextract",
        "@jextract//:bin/java",
        "@jextract//:files",
    ],
    outs = [
        "arrayfire-capi.jar",
    ],
    cmd = """
    HEADER=$(location @arrayfire//:include/arrayfire.h);
    $(location @jextract//:bin/jextract) \
      $(location @arrayfire//:include/arrayfire.h) \
      -I $$(dirname $$HEADER) \
      --define-macro AF_API_VERSION=39 \
      -t arrayfire.capi \
      ; \
      jar cf $(location arrayfire-capi.jar) arrayfire/capi/*;
    """,
)

java_import(
    name = "capi",
    jars = [
        ":arrayfire-capi.jar",
    ],
)

java_library(
    name = "arrayfire",
    srcs = glob(
        ["**/*.java"],
        exclude = ["**/*Test.java"],
    ),
    deps = [
        ":capi",
    ],
)

java_test(
    name = "ArrayFireTest",
    srcs = ["ArrayFireTest.java"],
    test_class = "arrayfire.ArrayFireTest",
    deps = [
        ":arrayfire",
    ],
)
