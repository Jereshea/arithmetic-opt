import os
import lit.formats
from lit.llvm import llvm_config

# 1. Suite name and shell test format (reads // RUN: lines)
config.name = "ARITHMETIC_OPT"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# 2. Only treat .mlir files as tests (ignores CMakeLists.txt, lit.cfg.py, etc.)
config.suffixes = [".mlir"]

# 3. Where test source files and build outputs live
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.arithmetic_obj_root, "test")

# 4. Standard %s, %t, %p substitutions
llvm_config.use_default_substitutions()

# 5. Tweak the PATH to include the tools dir.
tool_dirs = [config.arithmetic_tools_dir, config.llvm_tools_dir]
tools = ["arithmetic-opt", "FileCheck"]
llvm_config.add_tool_substitutions(tools, tool_dirs)