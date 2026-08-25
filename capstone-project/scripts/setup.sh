#!/usr/bin/env bash
set -Eeuo pipefail

LTX_REPOSITORY="https://github.com/Lightricks/LTX-2.git"
LTX_COMMIT="780984275fd47128b02bef9b5c085404276866ee"
readonly LTX_REPOSITORY LTX_COMMIT

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
LTX_CHECKOUT="${PROJECT_ROOT}/third_party/LTX-2"
PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv"
UV_CACHE="${PROJECT_ROOT}/.cache/uv"
readonly SCRIPT_DIR PROJECT_ROOT LTX_CHECKOUT PROJECT_ENVIRONMENT UV_CACHE

usage() {
    cat <<'EOF'
Usage: scripts/setup.sh

Checks the supported B200 build environment, checks out the pinned LTX-2
source, creates the repository-local Python environment, and builds the
custom attention extension.

Requirements:
  Linux x86-64, NVIDIA B200, CUDA toolkit 12.8, git, uv, and a C++ compiler.
  Python 3.12 and Ninja are installed into the repository environment.

The script does not install system packages or drivers. It refuses to alter
an existing dirty or non-official third_party/LTX-2 checkout.
EOF
}

die() {
    printf 'setup: error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf 'setup: %s\n' "$*"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_prerequisites() {
    local kernel_name machine_arch nvcc_output gpu_inventory gpu_line
    local found_b200=false

    kernel_name=$(uname -s)
    machine_arch=$(uname -m)
    [[ "${kernel_name}" == "Linux" ]] || die "Linux is required; found ${kernel_name}"
    [[ "${machine_arch}" == "x86_64" ]] || die "x86-64 is required; found ${machine_arch}"

    require_command git
    require_command uv
    require_command c++
    require_command nvcc
    require_command nvidia-smi

    nvcc_output=$(nvcc --version)
    [[ "${nvcc_output}" =~ release[[:space:]]12\.8 ]] || \
        die "CUDA toolkit 12.8 is required; nvcc reported: $(printf '%s\n' "${nvcc_output}" | tail -n 1)"

    gpu_inventory=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null) || \
        die "nvidia-smi could not query GPU name and compute capability"
    while IFS= read -r gpu_line; do
        gpu_line=${gpu_line//[[:space:]]/}
        if [[ "${gpu_line^^}" == *B200*,10.0 ]]; then
            found_b200=true
            break
        fi
    done <<<"${gpu_inventory}"
    [[ "${found_b200}" == true ]] || \
        die "an NVIDIA B200 with compute capability 10.0 is required; found: ${gpu_inventory//$'\n'/; }"

    log "prerequisites verified (CUDA 12.8, NVIDIA B200)"
}

official_remote() {
    case "$1" in
        https://github.com/Lightricks/LTX-2 | \
        https://github.com/Lightricks/LTX-2.git | \
        git@github.com:Lightricks/LTX-2.git | \
        ssh://git@github.com/Lightricks/LTX-2.git)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

require_clean_checkout() {
    local status
    status=$(git -C "${LTX_CHECKOUT}" status --porcelain --untracked-files=all)
    [[ -z "${status}" ]] || die \
        "${LTX_CHECKOUT} has local changes. Preserve or remove them before rerunning setup."
}

prepare_ltx_checkout() {
    local origin_url current_head current_branch

    if [[ ! -e "${LTX_CHECKOUT}" ]]; then
        log "cloning official LTX-2 source"
        mkdir -p -- "${PROJECT_ROOT}/third_party"
        GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-tags \
            "${LTX_REPOSITORY}" "${LTX_CHECKOUT}"
    fi

    [[ -d "${LTX_CHECKOUT}/.git" ]] || die \
        "${LTX_CHECKOUT} exists but is not a Git checkout; move it aside and rerun setup"

    origin_url=$(git -C "${LTX_CHECKOUT}" remote get-url origin 2>/dev/null) || \
        die "${LTX_CHECKOUT} has no origin remote"
    official_remote "${origin_url}" || die \
        "refusing non-official LTX-2 origin: ${origin_url}"
    require_clean_checkout

    if ! git -C "${LTX_CHECKOUT}" cat-file -e "${LTX_COMMIT}^{commit}" 2>/dev/null; then
        log "fetching pinned LTX-2 commit ${LTX_COMMIT}"
        GIT_LFS_SKIP_SMUDGE=1 git -C "${LTX_CHECKOUT}" fetch --depth=1 origin "${LTX_COMMIT}"
    fi

    current_head=$(git -C "${LTX_CHECKOUT}" rev-parse HEAD)
    current_branch=$(git -C "${LTX_CHECKOUT}" symbolic-ref -q --short HEAD || true)
    if [[ "${current_head}" != "${LTX_COMMIT}" || -n "${current_branch}" ]]; then
        log "checking out pinned LTX-2 commit ${LTX_COMMIT}"
        git -C "${LTX_CHECKOUT}" checkout --detach --quiet "${LTX_COMMIT}"
    fi

    current_head=$(git -C "${LTX_CHECKOUT}" rev-parse HEAD)
    [[ "${current_head}" == "${LTX_COMMIT}" ]] || die \
        "LTX-2 HEAD verification failed: expected ${LTX_COMMIT}, found ${current_head}"
    [[ -z "$(git -C "${LTX_CHECKOUT}" symbolic-ref -q --short HEAD || true)" ]] || die \
        "LTX-2 checkout must be detached at the pinned commit"
    require_clean_checkout
    log "LTX-2 checkout verified at ${LTX_COMMIT}"
}

sync_environment() {
    local -a sync_command

    [[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || die \
        "missing ${PROJECT_ROOT}/pyproject.toml"
    mkdir -p -- "${UV_CACHE}"

    sync_command=(uv sync --project "${PROJECT_ROOT}" --python 3.12)
    if [[ -f "${PROJECT_ROOT}/uv.lock" ]]; then
        sync_command+=(--frozen)
    fi

    log "syncing repository-local Python environment"
    UV_CACHE_DIR="${UV_CACHE}" \
    UV_PROJECT_ENVIRONMENT="${PROJECT_ENVIRONMENT}" \
        "${sync_command[@]}"
}

build_extension() {
    log "building the SM100 attention extension"
    UV_CACHE_DIR="${UV_CACHE}" \
    UV_PROJECT_ENVIRONMENT="${PROJECT_ENVIRONMENT}" \
        uv run --project "${PROJECT_ROOT}" --no-sync ltx-sm100 build
}

main() {
    case "${1:-}" in
        "")
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac
    [[ "$#" -le 1 ]] || die "setup does not accept positional arguments"

    check_prerequisites
    prepare_ltx_checkout
    sync_environment
    build_extension
    require_clean_checkout

    log "complete"
    log "environment: ${PROJECT_ENVIRONMENT}"
    log "LTX-2 source: ${LTX_CHECKOUT}"
}

main "$@"
