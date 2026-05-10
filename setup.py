from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1.0",
    description="OpenStack anomaly detection system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "anomaly-detect=anomaly_detection.cli:cli",
        ],
    },
)
