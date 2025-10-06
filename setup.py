"""
軍儀プロジェクトのセットアップスクリプト
"""

from setuptools import setup, find_packages

setup(
    name="gungi",
    version="1.0.0",
    description="軍儀 - ハンター×ハンターのボードゲーム実装",
    author="",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.12",
    install_requires=[
        "fastapi>=0.118.0",
        "uvicorn>=0.37.0",
        "pydantic>=2.11.10",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "ai": [
            "torch>=2.0.0",
            "numpy>=1.24.0",
        ],
    },
)
