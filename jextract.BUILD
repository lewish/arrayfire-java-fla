load("@bazel_skylib//rules:native_binary.bzl", "native_binary")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "files",
    srcs = glob(["**"])

)

# filegroup(
#     name = "lib_files",
#     srcs = glob(["lib/**/*.*"]),
# )

# native_binary(
#     name = "javac",
#     src = "bin/javac",
#     out = "javac",
#     data = [
#         ":lib_files",
#     ],
# )

# native_binary(
#     name = "java",
#     src = "bin/java",
#     out = "java",
#     data = [
#         ":lib_files",
#     ],
# )

exports_files(["bin/jextract", "bin/java"])

native_binary(
    name = "jextract",
    src = "bin/jextract",
    out = "bin/jextract",
    data = [
        ":files",
    ],
)
