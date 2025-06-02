from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="pdf-to-string",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python module that converts all types of PDF files to strings with special formatting for tables and other elements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aybek_jumashev/pdf-to-string",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "advanced": [
            "camelot-py[cv]>=0.10.1",
            "tabula-py>=2.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pdf-to-string=pdf_to_string:main",
        ],
    },
)