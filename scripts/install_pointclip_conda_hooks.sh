#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/home/zhangshuai/anaconda3}"
ENV_NAME="${ENV_NAME:-PointCLIP}"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Conda env not found: ${ENV_PREFIX}" >&2
  exit 1
fi

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/pointclip_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
export _POINTCLIP_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export POINTCLIP_SHOW_XFORMERS_WARNING="${POINTCLIP_SHOW_XFORMERS_WARNING:-0}"
EOF

cat > "${DEACTIVATE_DIR}/pointclip_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${_POINTCLIP_OLD_LD_LIBRARY_PATH+x}" ]]; then
  export LD_LIBRARY_PATH="${_POINTCLIP_OLD_LD_LIBRARY_PATH}"
  unset _POINTCLIP_OLD_LD_LIBRARY_PATH
fi
EOF

chmod +x "${ACTIVATE_DIR}/pointclip_cuda_paths.sh" "${DEACTIVATE_DIR}/pointclip_cuda_paths.sh"
echo "Installed PointCLIP conda activation hooks under ${ENV_PREFIX}/etc/conda"
