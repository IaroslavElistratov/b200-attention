#!/usr/bin/env bash
set -Eeuo pipefail

LTX_MODEL_REPOSITORY="Lightricks/LTX-2.3"
LTX_DISTILLED_FILENAME="ltx-2.3-22b-distilled-1.1.safetensors"
LTX_SPATIAL_UPSAMPLER_FILENAME="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_MODEL_REVISION="6f3520585aa27248020550da2f453aa0c572398c"
GEMMA_REPOSITORY="google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_REVISION="68f7ee4fbd59087436ada77ed2d62f373fdd4482"
readonly LTX_MODEL_REPOSITORY LTX_DISTILLED_FILENAME
readonly LTX_SPATIAL_UPSAMPLER_FILENAME LTX_MODEL_REVISION
readonly GEMMA_REPOSITORY GEMMA_REVISION

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
readonly SCRIPT_DIR PROJECT_ROOT HF_HOME
export HF_HOME

usage() {
    cat <<EOF
Usage: scripts/download_models.sh

Download all pinned model files required by the LTX-2.3 demo. Existing files
are reused, and interrupted downloads resume.

Files are stored under:
  models/checkpoints/${LTX_DISTILLED_FILENAME}
  models/checkpoints/${LTX_SPATIAL_UPSAMPLER_FILENAME}
  models/gemma/

Before the first run, accept the Gemma access terms and authenticate:
  HF_HOME="\$PWD/.cache/huggingface" .venv/bin/hf auth login
EOF
}

die() {
    printf 'download-models: error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf 'download-models: %s\n' "$*"
}

main() {
    case "$#" in
        0)
            ;;
        1)
            case "$1" in
                -h | --help)
                    usage
                    exit 0
                    ;;
                *)
                    usage >&2
                    die "unknown argument: $1"
                    ;;
            esac
            ;;
        *)
            usage >&2
            die "this script does not accept positional arguments"
            ;;
    esac

    local hf_bin="${PROJECT_ROOT}/.venv/bin/hf"
    local checkpoint_dir="${PROJECT_ROOT}/models/checkpoints"
    local gemma_dir="${PROJECT_ROOT}/models/gemma"
    local checkpoint_path="${checkpoint_dir}/${LTX_DISTILLED_FILENAME}"
    local spatial_upsampler_path="${checkpoint_dir}/${LTX_SPATIAL_UPSAMPLER_FILENAME}"

    [[ -x "${hf_bin}" ]] || die \
        "repository-local hf CLI not found; run scripts/setup.sh first"

    mkdir -p -- "${checkpoint_dir}" "${gemma_dir}"

    log "downloading pinned LTX-2.3 files"
    "${hf_bin}" download "${LTX_MODEL_REPOSITORY}" \
        "${LTX_DISTILLED_FILENAME}" "${LTX_SPATIAL_UPSAMPLER_FILENAME}" \
        --repo-type model --revision "${LTX_MODEL_REVISION}" \
        --local-dir "${checkpoint_dir}"
    [[ -s "${checkpoint_path}" ]] || die \
        "download completed without ${checkpoint_path}"
    [[ -s "${spatial_upsampler_path}" ]] || die \
        "download completed without ${spatial_upsampler_path}"

    log "downloading pinned Gemma text encoder"
    "${hf_bin}" download "${GEMMA_REPOSITORY}" \
        --repo-type model --revision "${GEMMA_REVISION}" \
        --local-dir "${gemma_dir}"
    [[ -s "${gemma_dir}/config.json" ]] || die \
        "download completed without ${gemma_dir}/config.json"

    log "downloads complete"
}

main "$@"
