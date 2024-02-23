package(default_visibility = ["//visibility:public"])

genrule(
    name = "version_include",
    outs = ["includes/af/version.h"],
    cmd = """
    touch $(location includes/af/version.h)
    """,
)

genrule(
    name = "compilers_include",
    outs = ["includes/af/compilers.h"],
    cmd = """
    touch $(location includes/af/compilers.h)
    """,
)

genrule(
    name = "generate_capi",
    srcs = [
        "@arrayfire//:include/arrayfire.h",
        ":version_include",
        ":compilers_include",
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
    VERSION_HEADER=$(location :version_include);
    VERSION_PARENT=$$(dirname $$VERSION_HEADER);
    $(location @jextract//:bin/jextract) \
      $(location @arrayfire//:include/arrayfire.h) \
      -I $$(dirname $$HEADER) \
      -I $$(dirname $$VERSION_PARENT) \
      --define-macro AF_API_VERSION=39 \
      --define-macro __CUDACC_RTC__=1 \
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
        ":capi",
    ],
)
