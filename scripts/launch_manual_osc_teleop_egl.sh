#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMMON_GIT_DIR="$(git -C "${SOURCE_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
if [[ "$(basename -- "${COMMON_GIT_DIR}")" == ".git" ]]; then
  CANONICAL_ROOT="$(dirname -- "${COMMON_GIT_DIR}")"
else
  CANONICAL_ROOT="${SOURCE_ROOT}"
fi
PROJECT_ROOT="$(dirname -- "${CANONICAL_ROOT}")"

PYTHON_BIN="${LIBERO_PYTHON:-${PROJECT_ROOT}/miniconda3/envs/libero/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "LIBERO Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

HOST_DRIVER_VERSION="$(
  awk '/^NVRM version:/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+([.][0-9]+)+$/) { print $i; exit } }' \
    /proc/driver/nvidia/version
)"
if [[ -z "${HOST_DRIVER_VERSION}" ]]; then
  echo "Cannot determine NVIDIA kernel driver version." >&2
  exit 1
fi

NVIDIA_BUNDLE_PARENT="${LIBERO_NVIDIA_BUNDLE_PARENT:-${CANONICAL_ROOT}/runtime/nvidia}"
NVIDIA_ROOT="${LIBERO_NVIDIA_RENDER_ROOT:-${NVIDIA_BUNDLE_PARENT}/${HOST_DRIVER_VERSION}}"
NVIDIA_LIB_DIR="${LIBERO_NVIDIA_USERSPACE_LIB_DIR:-${NVIDIA_ROOT}/runtime-libs-full}"
for required in \
  "${NVIDIA_LIB_DIR}/libcuda.so.1" \
  "${NVIDIA_LIB_DIR}/libEGL.so.1" \
  "${NVIDIA_ROOT}/10_nvidia.local.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "NVIDIA EGL bundle is incomplete: ${required}" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset LD_PRELOAD
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_ROOT}/10_nvidia.local.json"
export EGL_PLATFORM="surfaceless"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" scripts/run_manual_osc_teleop.py "$@"
