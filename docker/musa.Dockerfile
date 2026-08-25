# syntax=docker/dockerfile:1.7

# Moore Threads MUSA image for SGLang.
#   docker build -f docker/musa.Dockerfile -t sglang:musa .

ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG MUSA_APT_SOURCE=https://dl.mthreads.com/repo/repository/ubuntu2204/
ARG INSTALL_MUSA_STACK=auto
ARG MUSA_RUNTIME_VERSION=5.2
ARG MCCL_VERSION=2.4.0
ARG MUSA_MTML_VERSION=2.4.1
ARG MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.9.1.post1+musa5.2.0
ARG TORCH_MUSA_VERSION=2.9.1.post1+musa5.2.0
ARG TORCHAUDIO_VERSION=2.9.1+musa5.2.0
ARG TORCHVISION_VERSION=0.24.1.post1+musa5.2.0
ARG TRITON_VERSION=3.2.0
ARG TILELANG_MUSA_VERSION=0.1.8+musa.3
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_REF=main

ENV MUSA_HOME=/usr/local/musa \
    PATH=/root/.cargo/bin:/usr/local/mtshmem/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/mtshmem/lib:/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/lib \
    SGLANG_REPO_DIR=/workspace/sglang

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gcc \
        git \
        libsndfile1 \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python-is-python3 \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN printf 'deb [trusted=true] %s jammy main\n' "${MUSA_APT_SOURCE}" \
        > /etc/apt/sources.list.d/musa.list \
    && if [[ "${INSTALL_MUSA_STACK}" == "0" ]]; then \
        echo "Skipping MUSA apt stack install because INSTALL_MUSA_STACK=0"; \
        exit 0; \
    fi \
    && if [[ "${INSTALL_MUSA_STACK}" == "auto" ]] && command -v mcc >/dev/null 2>&1; then \
        echo "Keeping MUSA stack from BASE_IMAGE"; \
        mcc --version || true; \
        exit 0; \
    fi \
    && apt-get update \
    && if [[ "${MUSA_RUNTIME_VERSION}" =~ ^([0-9]+)\.([0-9]+)(\.|$) ]]; then \
        runtime_major="${BASH_REMATCH[1]}"; \
        runtime_minor="${BASH_REMATCH[2]}"; \
        runtime_suffix="${runtime_major}-${runtime_minor}"; \
    else \
        echo "MUSA_RUNTIME_VERSION must start with <major>.<minor>, got ${MUSA_RUNTIME_VERSION}" >&2; \
        exit 1; \
    fi \
    && resolve_apt_package() { \
        local logical="$1"; \
        local versions="$2"; \
        local allow_unversioned="$3"; \
        shift 3; \
        local spec=""; \
        local version pkg found_version; \
        for version in ${versions}; do \
            for pkg in "$@"; do \
                if ! apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    continue; \
                fi; \
                found_version="$(apt-cache madison "${pkg}" | awk -v w="${version}" '$3 ~ "^" w "([-+~:]|$)" && ex=="" {ex=$3} $3 ~ "^" w "[.]" && pf=="" {pf=$3} END {print (ex!="" ? ex : pf)}')"; \
                if [[ -n "${found_version}" ]]; then \
                    spec="${pkg}=${found_version}"; \
                    break 2; \
                fi; \
            done; \
        done; \
        if [[ -z "${spec}" && "${allow_unversioned}" == "1" ]]; then \
            for pkg in "$@"; do \
                if apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    spec="${pkg}"; \
                    break; \
                fi; \
            done; \
        fi; \
        if [[ -z "${spec}" ]]; then \
            echo "No apt package found for ${logical} with versions [${versions}]; checked: $*" >&2; \
            return 1; \
        fi; \
        echo "${spec}"; \
    } \
    && musa_pkg_defs=( \
        "musa-toolkit||musa-toolkit-${runtime_suffix}" \
        "musa-toolkit-config||musa-toolkit-${runtime_suffix}-config-common" \
        "mtcc||mtcc-${runtime_suffix}" \
        "musa-musart||musa-musart-${runtime_suffix}" \
        "musa-mupti||musa-mupti-${runtime_suffix}" \
        "musa-mualg||musa-mualg-${runtime_suffix}" \
        "musa-muthrust||musa-muthrust-${runtime_suffix}" \
        "libmublas||libmublas-${runtime_suffix}" \
        "libmufft||libmufft-${runtime_suffix}" \
        "libmupp||libmupp-${runtime_suffix}" \
        "libmurand||libmurand-${runtime_suffix}" \
        "libmusparse||libmusparse-${runtime_suffix}" \
        "libmusolver||libmusolver-${runtime_suffix}" \
        "libmublaslt||libmublaslt-${runtime_suffix}" \
        "libmthreads-compute|${MUSA_RUNTIME_VERSION}|libmthreads-compute" \
        "libmudnn3||libmudnn3-musa-${runtime_suffix},libmudnn3-musa-${runtime_major}" \
        "libmudnn3-dev||libmudnn3-dev-musa-${runtime_suffix},libmudnn3-musa-${runtime_major}-dev" \
        "libmthreads-mtml|${MUSA_MTML_VERSION}|libmthreads-mtml" \
        "mccl-s5000|${MCCL_VERSION}|mccl-s5000" \
    ) \
    && musa_specs=() \
    && for musa_def in "${musa_pkg_defs[@]}"; do \
        IFS='|' read -r musa_logical musa_versions musa_pkgs <<< "${musa_def}"; \
        IFS=',' read -r -a musa_pkg_arr <<< "${musa_pkgs}"; \
        if [[ -n "${musa_versions}" ]]; then musa_allow_unv=0; else musa_allow_unv=1; fi; \
        musa_spec="$(resolve_apt_package "${musa_logical}" "${musa_versions}" "${musa_allow_unv}" "${musa_pkg_arr[@]}")" || exit 1; \
        echo "Pinning ${musa_logical}: ${musa_spec}"; \
        musa_specs+=("${musa_spec}"); \
    done \
    && apt-get install -y --allow-downgrades --no-install-recommends "${musa_specs[@]}" \
    && printf '%s\n' \
        "${MUSA_HOME}/lib" \
        "${MUSA_HOME}/mudnn/lib" \
        "/usr/local/mtshmem/lib" \
        "/usr/lib/x86_64-linux-gnu" \
        > /etc/ld.so.conf.d/musa-runtime.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && cargo --version \
    && rustc --version

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-deps \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
        "torch==${TORCH_VERSION}" \
        "torch_musa==${TORCH_MUSA_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "triton==${TRITON_VERSION}" \
        "tilelang_musa==${TILELANG_MUSA_VERSION}" \
    && python3 -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        --extra-index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
        "torch==${TORCH_VERSION}" \
        "torch_musa==${TORCH_MUSA_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "triton==${TRITON_VERSION}" \
        "tilelang_musa==${TILELANG_MUSA_VERSION}" \
        "torchada>=0.1.84"

RUN git clone "${SGLANG_REPO}" "${SGLANG_REPO_DIR}" \
    && git -C "${SGLANG_REPO_DIR}" checkout --detach "${SGLANG_REF}"

WORKDIR ${SGLANG_REPO_DIR}

RUN cp python/pyproject_other.toml python/pyproject.toml \
    && python3 -m pip install -e "python[all_musa]" \
        --no-build-isolation \
        --index-url "${PYPI_INDEX_URL}" \
        --extra-index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
    && cd python/sglang/kernels/aot \
    && MTGPU_TARGET=mp_31 python3 setup_musa.py install

RUN python3 - <<'PY'
import torch

assert getattr(torch.version, "musa", None), "the PyTorch build is not MUSA-enabled"
assert hasattr(torch, "musa") and torch.musa.is_available(), "torch.musa is unavailable"
import torchada  # noqa: F401
import triton
assert triton.__version__ == "3.2.0", triton.__version__
import triton.backends.mtgpu  # noqa: F401
import tilelang  # noqa: F401
import sglang  # noqa: F401
PY

CMD ["/bin/bash"]
