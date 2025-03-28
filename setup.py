from setuptools import setup, find_packages

setup(
    name="insanely-fast-whisper",
    version="0.1.0",
    description="Optimized CLI tool for fast audio transcription using Whisper",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/insanely-fast-whisper",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "pyannote.audio>=3.1.0",
        "rich>=13.7.0",
        "numpy>=1.20.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "flash-attn>=2.0.0;platform_system!='Darwin'",
        ],
    },
    entry_points={
        "console_scripts": [
            "insanely-fast-whisper=whisper_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
