"""
Auto-download functionality for TRUE SOTA model weights.

Automatically downloads pretrained weights on first use, similar to HuggingFace.
"""

import os
import hashlib
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Callable
import structlog

logger = structlog.get_logger(__name__)

# Default cache directory
CACHE_DIR = Path(os.environ.get("AION_CACHE_DIR", os.path.expanduser("~/.cache/aion")))


def get_cache_dir(subdir: str = "") -> Path:
    """Get cache directory, creating if needed."""
    cache_dir = CACHE_DIR / subdir if subdir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(
    url: str,
    dest_path: Path,
    expected_hash: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """
    Download a file with optional hash verification.

    Args:
        url: URL to download from
        dest_path: Destination path
        expected_hash: Optional SHA256 hash for verification
        progress_callback: Optional callback(downloaded, total) for progress

    Returns:
        True if successful
    """
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading", url=url, dest=str(dest_path))

        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                progress_callback(downloaded, total_size)

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)

        # Verify hash if provided
        if expected_hash:
            actual_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                logger.error("Hash mismatch", expected=expected_hash, actual=actual_hash)
                dest_path.unlink()
                return False

        logger.info("Download complete", path=str(dest_path))
        return True

    except Exception as e:
        logger.error("Download failed", url=url, error=str(e))
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract zip or tar archive."""
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
        elif archive_path.suffix in (".tar", ".gz", ".tgz"):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(dest_dir)
        else:
            logger.warning("Unknown archive format", path=str(archive_path))
            return False
        return True
    except Exception as e:
        logger.error("Extraction failed", error=str(e))
        return False


# =============================================================================
# DNSMOS P.835 Weights
# =============================================================================

DNSMOS_URLS = {
    "sig_bak_ovr": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
    "model_v8": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx",
}


def get_dnsmos_model_path(model_name: str = "sig_bak_ovr") -> Optional[Path]:
    """
    Get DNSMOS ONNX model path, downloading if needed.

    Args:
        model_name: Model variant ("sig_bak_ovr" or "model_v8")

    Returns:
        Path to ONNX model file, or None if download failed
    """
    cache_dir = get_cache_dir("dnsmos")
    model_path = cache_dir / f"{model_name}.onnx"

    if model_path.exists():
        return model_path

    url = DNSMOS_URLS.get(model_name)
    if not url:
        logger.error("Unknown DNSMOS model", model=model_name)
        return None

    if download_file(url, model_path):
        return model_path
    return None


def download_all_dnsmos() -> dict[str, Optional[Path]]:
    """Download all DNSMOS models."""
    return {name: get_dnsmos_model_path(name) for name in DNSMOS_URLS}


# =============================================================================
# NISQA Weights
# =============================================================================

NISQA_URL = "https://github.com/gabrielmittag/NISQA/releases/download/v1.0/nisqa.tar.gz"
NISQA_WEIGHTS_URL = "https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar"


def get_nisqa_model_path() -> Optional[Path]:
    """
    Get NISQA model path, downloading if needed.

    Returns:
        Path to NISQA weights directory, or None if download failed
    """
    cache_dir = get_cache_dir("nisqa")
    weights_dir = cache_dir / "weights"
    marker_file = weights_dir / ".downloaded"

    if marker_file.exists():
        return weights_dir

    # Try to download weights
    archive_path = cache_dir / "nisqa_weights.tar"

    # Try primary URL first, then fallback
    for url in [NISQA_WEIGHTS_URL, NISQA_URL]:
        try:
            if download_file(url, archive_path):
                weights_dir.mkdir(parents=True, exist_ok=True)
                if extract_archive(archive_path, weights_dir):
                    marker_file.touch()
                    archive_path.unlink()  # Cleanup archive
                    return weights_dir
        except Exception as e:
            logger.warning(f"NISQA download attempt failed: {e}")
            continue

    logger.warning("NISQA weights download failed, using neural fallback")
    return None


# =============================================================================
# EINV2 / SELD Weights
# =============================================================================

# DCASE SELD baseline weights (publicly available)
SELD_URLS = {
    "seld_baseline": "https://zenodo.org/record/6387880/files/seld_baseline.zip",
}


def get_seld_model_path(model_name: str = "seld_baseline") -> Optional[Path]:
    """
    Get SELD model path, downloading if needed.

    Args:
        model_name: Model variant

    Returns:
        Path to model weights, or None if download failed
    """
    cache_dir = get_cache_dir("seld")
    model_dir = cache_dir / model_name
    marker_file = model_dir / ".downloaded"

    if marker_file.exists():
        return model_dir

    url = SELD_URLS.get(model_name)
    if not url:
        logger.info("SELD model not in registry, using neural architecture", model=model_name)
        return None

    archive_path = cache_dir / f"{model_name}.zip"

    if download_file(url, archive_path):
        model_dir.mkdir(parents=True, exist_ok=True)
        if extract_archive(archive_path, model_dir):
            marker_file.touch()
            archive_path.unlink()
            return model_dir

    return None


# =============================================================================
# StyleTTS2 Weights
# =============================================================================

STYLETTS2_URLS = {
    "ljspeech": "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth",
    "libritts": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth",
}


def get_styletts2_model_path(model_name: str = "ljspeech") -> Optional[Path]:
    """
    Get StyleTTS2 model path, downloading if needed.

    Args:
        model_name: Model variant ("ljspeech" or "libritts")

    Returns:
        Path to model weights, or None if download failed
    """
    cache_dir = get_cache_dir("styletts2")
    model_path = cache_dir / f"{model_name}.pth"

    if model_path.exists():
        return model_path

    url = STYLETTS2_URLS.get(model_name)
    if not url:
        logger.warning("Unknown StyleTTS2 model", model=model_name)
        return None

    if download_file(url, model_path):
        return model_path
    return None


# =============================================================================
# Utility: Download all weights
# =============================================================================

def download_all_weights(verbose: bool = True) -> dict[str, bool]:
    """
    Download all TRUE SOTA weights.

    Args:
        verbose: Print progress

    Returns:
        Dictionary of model -> success status
    """
    results = {}

    if verbose:
        print("Downloading TRUE SOTA model weights...")

    # DNSMOS
    if verbose:
        print("  [1/4] DNSMOS P.835...")
    dnsmos = download_all_dnsmos()
    results["dnsmos"] = all(v is not None for v in dnsmos.values())

    # NISQA
    if verbose:
        print("  [2/4] NISQA...")
    results["nisqa"] = get_nisqa_model_path() is not None

    # SELD
    if verbose:
        print("  [3/4] SELD baseline...")
    results["seld"] = get_seld_model_path() is not None

    # StyleTTS2
    if verbose:
        print("  [4/4] StyleTTS2...")
    results["styletts2"] = get_styletts2_model_path() is not None

    if verbose:
        print("\nResults:")
        for model, success in results.items():
            status = "✓" if success else "✗ (will use fallback)"
            print(f"  {model}: {status}")

    return results


if __name__ == "__main__":
    # Allow running as script to pre-download all weights
    download_all_weights(verbose=True)
