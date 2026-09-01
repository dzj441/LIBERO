#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${LIBERO_PYTHON:-${PROJECT_ROOT}/miniconda3/envs/libero/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "LIBERO Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set LIBERO_PYTHON to the configured environment's Python." >&2
  exit 1
fi

HOST_DRIVER_VERSION=""
if [[ -r /proc/driver/nvidia/version ]]; then
  HOST_DRIVER_VERSION="$(
    awk '/^NVRM version:/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+([.][0-9]+)+$/) { print $i; exit } }' \
      /proc/driver/nvidia/version
  )"
fi
if [[ -z "${HOST_DRIVER_VERSION}" ]]; then
  echo "Cannot determine the NVIDIA kernel-module version from /proc." >&2
  exit 1
fi

# Reuse the driver-matched symlink farm prepared for UniVTAC. The wrapper never
# copies driver libraries into the conda environment, so switching environments
# or upgrading the host driver cannot leave stale NVIDIA binaries behind.
NVIDIA_BUNDLE_PARENT="${LIBERO_NVIDIA_BUNDLE_PARENT:-${REPO_ROOT}/runtime/nvidia}"
if [[ ! -d "${NVIDIA_BUNDLE_PARENT}" ]]; then
  echo "NVIDIA bundle link is unavailable: ${NVIDIA_BUNDLE_PARENT}" >&2
  echo "Create runtime/nvidia as a symlink to the shared versioned driver directory," >&2
  echo "or set LIBERO_NVIDIA_BUNDLE_PARENT explicitly." >&2
  exit 1
fi
DEFAULT_NVIDIA_ROOT="${NVIDIA_BUNDLE_PARENT}/${HOST_DRIVER_VERSION}"
NVIDIA_ROOT="${LIBERO_NVIDIA_RENDER_ROOT:-${DEFAULT_NVIDIA_ROOT}}"
NVIDIA_ROOT="${NVIDIA_ROOT%/}"
NVIDIA_LIB_DIR="${LIBERO_NVIDIA_USERSPACE_LIB_DIR:-${NVIDIA_ROOT}/runtime-libs-full}"
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR%/}"

if [[ "$(basename -- "${NVIDIA_ROOT}")" != "${HOST_DRIVER_VERSION}" ]]; then
  echo "NVIDIA driver mismatch: bundle=$(basename -- "${NVIDIA_ROOT}"), kernel=${HOST_DRIVER_VERSION}" >&2
  exit 1
fi

for required_path in \
  "${NVIDIA_LIB_DIR}/libcuda.so.1" \
  "${NVIDIA_LIB_DIR}/libEGL.so.1" \
  "${NVIDIA_LIB_DIR}/libEGL_nvidia.so.0" \
  "${NVIDIA_LIB_DIR}/libnvidia-eglcore.so.${HOST_DRIVER_VERSION}" \
  "${NVIDIA_ROOT}/10_nvidia.local.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "NVIDIA EGL bundle is incomplete: ${required_path}" >&2
    exit 1
  fi
done

# Exclude host-injected driver directories that could shadow part of the
# selected bundle, while preserving unrelated application/CUDA locations.
SANITIZED_LIBRARY_PATH=""
IFS=':' read -r -a INHERITED_LIBRARY_PATHS <<< "${LD_LIBRARY_PATH:-}"
for library_path in "${INHERITED_LIBRARY_PATHS[@]}"; do
  [[ -z "${library_path}" ]] && continue
  case "${library_path}" in
    /usr/local/nvidia|/usr/local/nvidia/*|/usr/local/cuda/compat|/usr/local/cuda/compat/*)
      continue
      ;;
  esac
  if [[ -z "${SANITIZED_LIBRARY_PATH}" ]]; then
    SANITIZED_LIBRARY_PATH="${library_path}"
  else
    SANITIZED_LIBRARY_PATH="${SANITIZED_LIBRARY_PATH}:${library_path}"
  fi
done

export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}${SANITIZED_LIBRARY_PATH:+:${SANITIZED_LIBRARY_PATH}}"
unset LD_PRELOAD
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_ROOT}/10_nvidia.local.json"
export __GLX_VENDOR_LIBRARY_NAME="nvidia"
export EGL_PLATFORM="surfaceless"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

cd "${REPO_ROOT}"
TARGET_SCRIPT="scripts/replay_demonstration.py"
if [[ "${1:-}" == "--python-script" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--python-script requires a repository-relative Python path" >&2
    exit 1
  fi
  TARGET_SCRIPT="$2"
  shift 2
fi
exec "${PYTHON_BIN}" "${TARGET_SCRIPT}" "$@"
