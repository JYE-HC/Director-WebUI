from __future__ import annotations

import os

import uvicorn


def run() -> None:
    uvicorn.run(
        "director.app:app",
        host=os.getenv("DIRECTOR_HOST", "127.0.0.1"),
        port=int(os.getenv("DIRECTOR_PORT", "8787")),
        reload=False,
    )


if __name__ == "__main__":
    run()
