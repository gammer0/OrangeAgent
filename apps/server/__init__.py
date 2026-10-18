"""Harness web service layer (FastAPI).

Boundary with the core ``harness`` package:

* ``harness``: the orchestration kernel with zero web-framework dependencies;
* ``apps.server``: wraps the kernel as a REST + SSE service.
"""
