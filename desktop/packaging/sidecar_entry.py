"""PyInstaller entrypoint for the KnowledgeAgent sidecar.

Kept outside `backend/app/` so PyInstaller does not add `backend/app/` to
`sys.path`, which would otherwise shadow the third-party `mcp` SDK package
with `backend/app/mcp/`.
"""

import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse-pdf-worker"]:
        from app.parsing.pdf import pdf_worker_main

        raise SystemExit(pdf_worker_main(sys.argv[2:]))
    if sys.argv[1:2] == ["--parse-docx-worker"]:
        from app.parsing.docx import docx_worker_main

        raise SystemExit(docx_worker_main(sys.argv[2:]))
    from app.sidecar import main

    raise SystemExit(main())
