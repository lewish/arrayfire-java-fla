load("@aspect_rules_ts//ts:defs.bzl", native_ts_library = "ts_project")
# load("@build_bazel_rules_nodejs//:index.bzl", "nodejs_binary")

def ts_library(tsconfig = None, **kwargs):
    native_ts_library(
        tsconfig = tsconfig if tsconfig else "//:tsconfig",
        declaration = True,
        source_map = True,
        **kwargs
    )

# def ts_binary(deps = [], templated_args = [], **kwargs):
#     nodejs_binary(
#         data = deps + [
#             "@npm//source-map-support",
#         ],
#         link_workspace_root = True,
#         source_map = True,
#         templated_args = templated_args + ["--node_options=--require=source-map-support/register"],
#         **kwargs
#     )
