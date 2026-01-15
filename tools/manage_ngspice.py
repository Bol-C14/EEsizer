#!/usr/bin/env python3
"""
Manage ngspice dependency for offline/local usage.
Supports downloading .deb packages and extracting them for user-local installation.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Debian Bookworm (Stable) versions - generally good compatibility
NGSPICE_VERSION = "39.3+ds-1"
BASE_URL = "http://ftp.debian.org/debian/pool/main/n/ngspice"

# Map python platform.machine() to Debian arch
ARCH_MAP = {
    "x86_64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

PROJECT_ROOT = Path(__file__).parent.parent
OFFLINE_DEPS_DIR = PROJECT_ROOT / "tools" / "offline_deps"
VENDOR_DIR = PROJECT_ROOT / "vendor"
INSTALL_DIR = VENDOR_DIR / "ngspice"


def get_arch():
    machine = platform.machine()
    return ARCH_MAP.get(machine, machine)


def normalize_arch(arch: str) -> str:
    """Normalize user-provided arch keys to Debian arch names.

    Accepts keys like 'x86_64', 'aarch64', or already-normalized names like 'amd64'.
    """
    if arch in ARCH_MAP:
        return ARCH_MAP[arch]
    if arch in ARCH_MAP.values():
        return arch
    # common aliases
    if arch in ("x86-64", "x64", "x86"):
        return "amd64"
    if arch in ("aarch64", "arm64"):
        return "arm64"
    return arch


def get_deb_filename(arch):
    return f"ngspice_{NGSPICE_VERSION}_{arch}.deb"


def get_download_url(filename):
    return f"{BASE_URL}/{filename}"


def download_package(arch, force=False):
    arch_norm = normalize_arch(arch)
    filename = get_deb_filename(arch_norm)
    url = get_download_url(filename)
    dest_path = OFFLINE_DEPS_DIR / filename

    if dest_path.exists() and not force:
        print(f"Package already exists at {dest_path}")
        return dest_path

    print(f"Downloading {url}...")
    OFFLINE_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to {dest_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)
    return dest_path


def extract_deb(deb_path, install_dir):
    """
    Extracts a .deb file into install_dir.
    Roughly equivalent to: ar x file.deb; tar xf data.tar.xz -C install_dir
    """
    print(f"Extracting {deb_path} to {install_dir}...")
    
    # We need 'ar' and 'tar'
    if not shutil.which("ar"):
        print("Error: 'ar' utility not found. Please install binutils.")
        sys.exit(1)
    if not shutil.which("tar"):
        print("Error: 'tar' utility not found.")
        sys.exit(1)

    install_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary work directory
    work_dir = install_dir / ".tmp_extract"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    try:
        # 1. Extract .deb using ar (extracts control.tar.xz, data.tar.xz, debian-binary)
        subprocess.check_call(["ar", "x", str(deb_path.resolve())], cwd=work_dir)
        
        # 2. Find data archive (usually data.tar.xz or data.tar.gz)
        data_archive = None
        for f in work_dir.iterdir():
            if f.name.startswith("data.tar"):
                data_archive = f
                break
        
        if not data_archive:
            print("Error: Could not find data.tar.* in .deb package")
            sys.exit(1)

        # 3. Extract data archive to install_dir
        # We strip components corresponding to absolute paths (usr/bin -> bin) if we want a flatter structure,
        # but keeping the structure is safer for libs.
        # Actually, let's keep the structure: {install_dir}/usr/bin/ngspice
        subprocess.check_call(["tar", "-xf", data_archive.name], cwd=work_dir)
        
        # Move contents from work_dir to install_dir
        # The tarball likely contains ./usr/bin/... or usr/bin/...
        # We'll validte what we got.
        
        # Common pattern: usr/bin/ngspice
        usr_dir = work_dir / "usr"
        if usr_dir.exists():
            # Merge into install_dir
            cmd = f"cp -r {usr_dir}/* {install_dir}/"
            # Python cp -r is tricky with existing dirs, using shell cp for simplicity in this script
            subprocess.check_call(f"cp -r {usr_dir}/* {install_dir}/", shell=True)
        else:
            print("Warning: Unexpected archive structure (no 'usr' folder found). contents:")
            subprocess.check_call(["ls", "-R"], cwd=work_dir)

    finally:
        shutil.rmtree(work_dir)
    
    print("Extraction complete.")


def cleanup():
    if INSTALL_DIR.exists():
        shutil.rmtree(INSTALL_DIR)
    print(f"Removed {INSTALL_DIR}")


def verify_installation():
    # Check for binary
    bin_path = INSTALL_DIR / "bin" / "ngspice"
    if bin_path.exists():
        print(f"ngspice binary found at: {bin_path}")
        # Try to run version check
        try:
            # Set library path if needed (though we aren't downloading libs, just the main package)
            # This might fail if system libs are missing.
            res = subprocess.run([str(bin_path), "-v"], capture_output=True, text=True)
            print("Version check output:")
            print(res.stdout or res.stderr)
        except Exception as e:
            print(f"Warning: Could not run binary to verify version: {e}")
    else:
        print(f"Error: binary not found at {bin_path}")


def main():
    parser = argparse.ArgumentParser(description="Manage local ngspice installation")
    parser.add_argument("--download", action="store_true", help="Download package for current arch")
    parser.add_argument("--install", action="store_true", help="Install (extract) from offline package")
    parser.add_argument("--clean", action="store_true", help="Remove installed vendor directory")
    parser.add_argument("--arch", default=get_arch(), help="Target architecture (default: current)")
    parser.add_argument("--force", action="store_true", help="Force download/install")

    args = parser.parse_args()

    ensure_download = args.download
    do_install = args.install
    
    # Default behavior if flags are mixed or missing?
    # User said: "download with network... then keep... then install"
    # If no args, maybe just status check?
    if not (args.download or args.install or args.clean):
        parser.print_help()
        return

    if args.clean:
        cleanup()

    arch_norm = normalize_arch(args.arch)
    deb_path = OFFLINE_DEPS_DIR / get_deb_filename(arch_norm)

    if ensure_download:
        download_package(args.arch, force=args.force)

    if do_install:
        if not deb_path.exists():
            print(f"Package {deb_path} not found.")
            if ensure_download: # Should have been downloaded above, unless failed
                print("Download seemed to fail?")
            else:
                print("Run with --download first or ensure file is present.")
            sys.exit(1)
        
        extract_deb(deb_path, INSTALL_DIR)
        verify_installation()


if __name__ == "__main__":
    main()
