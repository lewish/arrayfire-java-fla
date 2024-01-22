load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
def arrayfire_java_fla_deps():
    maybe(
        http_archive,
        name = "jextract",
        sha256 = "83626610b1b074bfe4985bd825d8ba44d906a30b24c42d971b6ac836c7eb0671",
        urls = ["https://download.java.net/java/early_access/jextract/1/openjdk-21-jextract+1-2_linux-x64_bin.tar.gz"],
        build_file = "@arrayfire_java_fla//:jextract.BUILD",
        strip_prefix = "jextract-21",
    )
    maybe(
        http_archive,
        name = "arrayfire",
        sha256 = "ffd078dde66a1a707d049f5d2dab128e86748a92ca7204d0b3a7933a9a9904be",
        urls = ["https://github.com/arrayfire/arrayfire/archive/refs/tags/v3.9.0.tar.gz"],
        build_file = "@arrayfire_java_fla//:arrayfire.BUILD",
        strip_prefix = "arrayfire-3.9.0",
    )
