"""SQL scope shared by the local retrieval services."""

from collections.abc import Collection, Sequence

MAX_FILTER_IDS = 128
CURRENT_VERSION = """ AND v.rowid = (SELECT max(revision.rowid) FROM document_versions AS revision
    WHERE revision.document_id = v.document_id)"""
LATEST_PER_DOCUMENT = """SELECT * FROM (
    SELECT *, row_number() OVER (
        PARTITION BY document_id ORDER BY version_order DESC, chunk_set_id DESC
    ) AS preference FROM eligible
) WHERE preference = 1"""


def scope_filters(document_ids: Sequence[str] | None, chunk_set_ids: Sequence[int] | None, *,
                  label: str) -> tuple[str, tuple]:
    clauses, values = [], []
    for column, ids in (("v.document_id", document_ids), ("s.chunk_set_id", chunk_set_ids)):
        if ids is None:
            continue
        if not isinstance(ids, Collection) or isinstance(ids, (str, bytes)) or len(ids) > MAX_FILTER_IDS:
            raise ValueError(f"Invalid {label} filter size.")
        for value in ids:
            valid = (
                isinstance(value, str) and 0 < len(value) <= 128
                and "\x00" not in value and not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
                if column == "v.document_id" else
                type(value) is int and 1 <= value <= 2**63 - 1
            )
            if not valid:
                raise ValueError(f"Invalid {label} filter value for {column}.")
        placeholders = ", ".join("?" for _ in ids)
        clauses.append(f" AND {column} IN ({placeholders})" if ids else " AND 0")
        values.extend(ids)
    return "".join(clauses), tuple(values)
