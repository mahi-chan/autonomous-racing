from setuptools import setup, find_packages

setup(
    name="f1-racing-rl",
    version="1.0.0",
    author="F1 Racing RL Team",
    description="State-of-the-art RL system for F1 racing optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # Core RL
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.2.0",
        "sb3-contrib>=2.2.0",

        # Physics simulation
        "mujoco>=3.0.0",
        "pybullet>=3.2.5",

        # Deep learning
        "torch>=2.0.0",
        "tensorboard>=2.14.0",

        # Distributed training
        "ray[rllib]>=2.8.0",

        # Scientific computing
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",

        # Visualization
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "seaborn>=0.12.0",

        # Data handling
        "h5py>=3.9.0",
        "pyyaml>=6.0",

        # Logging
        "wandb>=0.15.0",
        "mlflow>=2.8.0",

        # Utilities
        "tqdm>=4.66.0",
        "click>=8.1.0",
        "rich>=13.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "pyvista>=0.42.0",
            "open3d>=0.17.0",
            "dash>=2.13.0",
            "streamlit>=1.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "f1-train=scripts.train:main",
            "f1-eval=scripts.evaluate:main",
            "f1-viz=scripts.visualize:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
