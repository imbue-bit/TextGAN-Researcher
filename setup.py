from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deep-research-agent",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="一个基于TextGAN-D架构的智能研究代理，能够进行深入、批判性的研究并生成高质量报告",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deep-research-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "httpx>=0.25.0",
            "python-multipart>=0.0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-research=examples.usage_example:main",
            "deep-research-api=api.run_api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai research agent textgan deep-learning langchain openai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/deep-research-agent/issues",
        "Source": "https://github.com/yourusername/deep-research-agent",
        "Documentation": "https://github.com/yourusername/deep-research-agent#readme",
    },
) 