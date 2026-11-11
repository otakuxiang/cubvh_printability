"""Reproducible OptiX device PTX + PyTorch extension build: python -m cubvh_printability.build_optix."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess


@lru_cache(maxsize=1)
def build(verbose=False, require_device=True):
    from .core import PrintabilityDependencyError
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError as exc:
        raise PrintabilityDependencyError('OptiX requires CUDA-enabled PyTorch') from exc
    sdk = Path(os.environ.get('OPTIX_ROOT', '')) / 'include'
    if not (sdk / 'optix.h').is_file() or not CUDA_HOME:
        raise PrintabilityDependencyError('Set OPTIX_ROOT to OptiX SDK >=7.4 and CUDA_HOME to a CUDA toolkit with nvcc')
    if require_device and not torch.cuda.is_available():
        raise PrintabilityDependencyError('OptiX requires an NVIDIA GPU and CUDA-enabled PyTorch')
    source = Path(__file__).with_name('optix_src')
    nvcc = str(Path(CUDA_HOME) / 'bin' / 'nvcc')
    digest = hashlib.sha256()
    for path in sorted(source.iterdir()):
        if path.is_file():
            digest.update(path.read_bytes())
    digest.update((sdk / 'optix.h').read_bytes())
    digest.update(subprocess.check_output([nvcc, '--version']))
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', Path.home() / '.cache' / 'torch_extensions')) / ('cubvh_optix_' + digest.hexdigest()[:16])
    cache.mkdir(parents=True, exist_ok=True)
    # Use a process-unique intermediate; atomic replace allows concurrent builds.
    ptx = cache / 'trace.ptx'
    if not ptx.exists():
        temporary = cache / f'trace.{os.getpid()}.ptx'
        try:
            subprocess.run([nvcc, '--ptx', '-std=c++17', '-O3', '-arch=compute_70',
                            '-I' + str(sdk), str(source / 'trace.cu'), '-o', str(temporary)], check=True)
            temporary.replace(ptx)
        finally:
            temporary.unlink(missing_ok=True)
    module = load(name='cubvh_optix', sources=[str(source / 'bindings.cpp')],
                  extra_include_paths=[str(sdk)], extra_cflags=['-O3', '-std=c++17'],
                  extra_ldflags=['-ldl'], with_cuda=True, build_directory=str(cache), verbose=verbose)
    return module, ptx.read_text()


if __name__ == '__main__':
    build(verbose=True, require_device=False)
