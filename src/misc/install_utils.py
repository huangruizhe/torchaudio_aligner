"""
Installation utilities for torchaudio_aligner dependencies.

This module provides helper functions to install k2 and other dependencies
with automatic version detection for PyTorch and CUDA compatibility.

Usage:
    from misc.install_utils import install_k2_if_needed, install_other_deps

    install_k2_if_needed()
    install_other_deps()

Or in a notebook:
    from misc.install_utils import install_all
    install_all()
"""

import subprocess
import sys
import re
import urllib.request
from typing import Optional, List, Tuple


def get_torch_info() -> Tuple[str, str, bool, Optional[str]]:
    """
    Get PyTorch version and CUDA information.

    Returns:
        Tuple of (torch_version, torch_major_minor, cuda_available, cuda_version)
    """
    import torch

    torch_version = torch.__version__.split('+')[0]  # e.g., "2.5.0"
    torch_major_minor = '.'.join(torch_version.split('.')[:2])  # e.g., "2.5"
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None

    return torch_version, torch_major_minor, cuda_available, cuda_version


def check_k2_installed() -> bool:
    """Check if k2 is already installed."""
    try:
        import k2
        return True
    except ImportError:
        return False


def get_k2_version() -> Optional[str]:
    """Get installed k2 version, or None if not installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "k2"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return None
    except Exception:
        return None


def find_matching_k2_version(
    index_url: str,
    pattern: str,
    verbose: bool = True,
) -> Optional[str]:
    """
    Find a matching k2 version from the k2 package index.

    Args:
        index_url: URL to the k2 package index (cuda.html or cpu.html)
        pattern: Regex pattern to match package names
        verbose: Print progress messages

    Returns:
        Package name if found, None otherwise
    """
    try:
        with urllib.request.urlopen(index_url, timeout=10) as response:
            html = response.read().decode('utf-8')

        matches = re.findall(pattern, html)

        if matches:
            # Get the latest version (last match)
            pkg_name = matches[-1].replace('k2-', 'k2==')
            if verbose:
                print(f"Found: {pkg_name}")
            return pkg_name
        return None
    except Exception as e:
        if verbose:
            print(f"Could not fetch index: {e}")
        return None


def install_k2_if_needed(verbose: bool = True) -> None:
    """
    Check if k2 is available, if not, install the correct version.

    Automatically detects PyTorch version and CUDA availability to install
    the matching k2 version.

    Args:
        verbose: Print progress messages

    Raises:
        RuntimeError: If k2 installation fails
    """
    # Check if already installed
    if check_k2_installed():
        if verbose:
            version = get_k2_version()
            print(f"k2 already installed: {version}")
        return

    # Get system info
    torch_version, torch_major_minor, cuda_available, cuda_version = get_torch_info()

    if verbose:
        print(f"PyTorch: {torch_version}")
        print(f"CUDA available: {cuda_available}")
        if cuda_version:
            print(f"CUDA version: {cuda_version}")

    # Determine which k2 to install
    if cuda_available and cuda_version:
        # GPU version
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])  # e.g., "12.4"
        index_url = "https://k2-fsa.github.io/k2/cuda.html"

        if verbose:
            print(f"\nLooking for k2 with CUDA {cuda_major_minor} and PyTorch {torch_major_minor}...")

        # Pattern: k2-1.24.4.dev20251030+cuda12.4.torch2.5.0
        pattern = rf'k2-[\d.]+dev\d+\+cuda{re.escape(cuda_major_minor)}\.torch{re.escape(torch_major_minor)}\.\d+'
        pkg_name = find_matching_k2_version(index_url, pattern, verbose)

        if pkg_name:
            cmd = [sys.executable, "-m", "pip", "install", pkg_name, "-f", index_url]
        else:
            if verbose:
                print(f"No exact match found for CUDA {cuda_major_minor} + PyTorch {torch_major_minor}")
                print("Trying generic GPU install...")
            cmd = [sys.executable, "-m", "pip", "install", "k2", "-f", index_url]
    else:
        # CPU version
        index_url = "https://k2-fsa.github.io/k2/cpu.html"

        if verbose:
            print(f"\nLooking for k2 CPU version for PyTorch {torch_major_minor}...")

        # Pattern: k2-1.24.4.dev20251030+cpu.torch2.5.0
        pattern = rf'k2-[\d.]+dev\d+\+cpu\.torch{re.escape(torch_major_minor)}\.\d+'
        pkg_name = find_matching_k2_version(index_url, pattern, verbose)

        if pkg_name:
            cmd = [sys.executable, "-m", "pip", "install", pkg_name, "--no-deps", "-f", index_url]
        else:
            if verbose:
                print(f"No exact match found for PyTorch {torch_major_minor}")
            cmd = [sys.executable, "-m", "pip", "install", "k2", "--no-deps", "-f", index_url]

    # Run installation
    if verbose:
        print(f"\nInstalling: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print("k2 installed successfully!")
        return
    else:
        error_msg = f"k2 installation failed: {result.stderr}"
        if verbose:
            print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)


def install_other_deps(
    deps: Optional[List[str]] = None,
    quiet: bool = True,
    verbose: bool = True,
) -> None:
    """
    Install other required dependencies.

    Args:
        deps: List of dependencies to install. If None, uses default list.
        quiet: Use pip's quiet mode (-q)
        verbose: Print progress messages

    Raises:
        RuntimeError: If any dependency installation fails
    """
    if deps is None:
        deps = [
            "pytorch-lightning",
            "cmudict",
            "g2p_en",
            "pydub",
            "pypdf",
            "requests",
            "beautifulsoup4",
            "torchcodec",
            "git+https://github.com/huangruizhe/lis.git",
        ]

    failed_deps = []
    for dep in deps:
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if quiet:
                cmd.append("-q")
            cmd.append(dep)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if verbose:
                    print(f"ERROR: Failed to install {dep}: {result.stderr}")
                failed_deps.append(dep)
        except Exception as e:
            if verbose:
                print(f"ERROR: Failed to install {dep}: {e}")
            failed_deps.append(dep)

    if failed_deps:
        raise RuntimeError(f"Failed to install dependencies: {', '.join(failed_deps)}")


def install_ocr_deps(verbose: bool = True) -> None:
    """
    Install OCR dependencies (pytesseract, pdf2image).

    Note: Also requires system packages:
        - Linux: apt install tesseract-ocr poppler-utils
        - macOS: brew install tesseract poppler

    Args:
        verbose: Print progress messages

    Raises:
        RuntimeError: If installation fails
    """
    deps = ["pytesseract", "pdf2image"]

    if verbose:
        print("Installing OCR dependencies...")
        print("Note: Also install system packages:")
        print("  Linux: apt install tesseract-ocr poppler-utils")
        print("  macOS: brew install tesseract poppler")

    install_other_deps(deps, quiet=True, verbose=verbose)


def install_romanization_deps(verbose: bool = True) -> None:
    """
    Install romanization dependencies for non-Latin scripts.

    Args:
        verbose: Print progress messages

    Raises:
        RuntimeError: If installation fails
    """
    if verbose:
        print("Installing romanization dependencies...")

    install_other_deps(["uroman-python"], quiet=True, verbose=verbose)


def install_all(verbose: bool = True) -> None:
    """
    Install all dependencies (k2 + other deps).

    This is the main entry point for setting up the environment.

    Args:
        verbose: Print progress messages

    Raises:
        RuntimeError: If any installation fails
    """
    if verbose:
        print("=" * 60)
        print("Installing torchaudio_aligner dependencies")
        print("=" * 60)

    # Install k2 (raises RuntimeError on failure)
    install_k2_if_needed(verbose=verbose)

    # Install other deps (raises RuntimeError on failure)
    if verbose:
        print("\nInstalling other dependencies...")
    install_other_deps(verbose=verbose)

    if verbose:
        print("\nDependency installation complete.")


# For convenience when running as script
if __name__ == "__main__":
    install_all()
