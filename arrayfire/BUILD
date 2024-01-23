load("//bazel:java.bzl", "java_library_auto", "java_test_auto")

package(default_visibility = ["//visibility:public"])

LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

java_library(
    name = "numbers",
    srcs = [
        letter + ".java"
        for letter in LETTERS
    ],
)

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

java_library_auto(
    recursive = True,
    deps = [
        ":capi",
    ],
)

java_test_auto(
    recursive = True,
)
