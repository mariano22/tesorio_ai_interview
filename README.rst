======
README
======

NOTE: An ``OPENAI_API_KEY`` will need to be set,
one will be provided for you.

Prerequisites
=============

- Docker

Health Check
============

One can verify this image is working as expected
by 1) building it via:

``docker build -t genai-tech-screen .``

and 2) executing it via

``docker run -e OPENAI_API_KEY=<key> -i genai-tech-screen``

If all goes well a sample dataframe shall be sent to OpenAI
for a sample response.

Project Structure
=================

- ``app/``: Sample module to build on (e.g. ``from app import prompts``)
- ``data/``: Directory containing data files.
- ``hello.py``: A basic health check script for the project setup.
- ``Dockerfile``: Defines the container image.
- ``requirements.txt``: Lists Python dependencies.

