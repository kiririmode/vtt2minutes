# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Get the project root directory
import os
project_root = Path(os.getcwd())
src_dir = project_root / "src"

# Add src directory to Python path
sys.path.insert(0, str(src_dir))

a = Analysis(
    [str(src_dir / "vtt2minutes" / "__main__.py")],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[
        # Include default templates and configuration files
        (str(project_root / "prompt_templates"), "prompt_templates"),
        (str(project_root / "filter_words.txt"), "."),
        (str(project_root / "replacement_rules.txt"), "."),
    ],
    hiddenimports=[
        # Ensure all vtt2minutes modules are included
        "vtt2minutes",
        "vtt2minutes.cli",
        "vtt2minutes.parser",
        "vtt2minutes.preprocessor", 
        "vtt2minutes.intermediate",
        "vtt2minutes.bedrock",
        # Include boto3 and related AWS dependencies
        "boto3",
        "botocore",
        "urllib3",
        # Include rich dependencies
        "rich",
        "rich.console",
        "rich.panel",
        "rich.progress",
        "rich.text",
        # Include click dependencies
        "click",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test modules and development tools
        "pytest",
        "pytest_cov",
        "pytest_asyncio",
        "ruff",
        "pyright",
        "pip_audit",
        "pyinstaller",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="vtt2minutes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)