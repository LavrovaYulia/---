from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="heart-attack-prediction-api",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="API для предсказания риска сердечного приступа",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/heart-attack-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "joblib>=1.3.2",
    ],
)