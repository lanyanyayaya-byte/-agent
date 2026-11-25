from app.storage.keyword_terms import TOKENIZER

# Both publication and the high-water mark belong to the same transaction as each batch.
INDEX_STORAGE = (
    """CREATE TABLE keyword_index_sets (
        chunk_set_id INTEGER PRIMARY KEY REFERENCES chunk_sets(chunk_set_id) ON DELETE CASCADE,
        last_chunk_index INTEGER NOT NULL DEFAULT -1 CHECK (last_chunk_index >= -1),
        indexed_at TEXT
    )""",
    """CREATE TABLE keyword_index_rows (
        row_id INTEGER PRIMARY KEY,
        chunk_set_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        UNIQUE (chunk_set_id, chunk_index),
        FOREIGN KEY (chunk_set_id, chunk_index) REFERENCES chunks(chunk_set_id, chunk_index) ON DELETE CASCADE
    )""",
    """CREATE VIEW keyword_index_content AS
       SELECT r.row_id, knowledge_keyword_terms(c.text) AS body,
              knowledge_keyword_title(c.title_context) AS title
       FROM keyword_index_rows AS r JOIN chunks AS c USING (chunk_set_id, chunk_index)""",
    f"""CREATE VIRTUAL TABLE keyword_fts USING fts5(body, title,
           content='keyword_index_content', content_rowid='row_id', tokenize="{TOKENIZER}")""",
)

INDEX_TRIGGERS = {
    "keyword_row_insert": """AFTER INSERT ON keyword_index_rows BEGIN
        INSERT INTO keyword_fts(rowid, body, title)
        SELECT row_id, body, title FROM keyword_index_content WHERE row_id = NEW.row_id;
    END""",
    "keyword_row_delete": """BEFORE DELETE ON keyword_index_rows BEGIN
        INSERT INTO keyword_fts(keyword_fts, rowid, body, title)
        SELECT 'delete', row_id, body, title FROM keyword_index_content WHERE row_id = OLD.row_id;
    END""",
    # Delete the mapping before the old text becomes inaccessible through the external-content view.
    "keyword_chunk_delete": """BEFORE DELETE ON chunks BEGIN
        DELETE FROM keyword_index_rows WHERE chunk_set_id = OLD.chunk_set_id AND chunk_index = OLD.chunk_index;
    END""",
    "keyword_chunk_insert": """AFTER INSERT ON chunks
        WHEN EXISTS (SELECT 1 FROM keyword_index_sets WHERE chunk_set_id = NEW.chunk_set_id
                     AND (indexed_at IS NOT NULL OR last_chunk_index >= NEW.chunk_index))
    BEGIN
        INSERT INTO keyword_index_rows(chunk_set_id, chunk_index) VALUES (NEW.chunk_set_id, NEW.chunk_index);
    END""",
    "keyword_chunk_update": """AFTER UPDATE OF text, title_context ON chunks
        WHEN NEW.text IS NOT OLD.text OR NEW.title_context IS NOT OLD.title_context
    BEGIN
        INSERT INTO keyword_fts(keyword_fts, rowid, body, title)
        SELECT 'delete', row_id, knowledge_keyword_terms(OLD.text), knowledge_keyword_title(OLD.title_context)
        FROM keyword_index_rows WHERE chunk_set_id = OLD.chunk_set_id AND chunk_index = OLD.chunk_index;
        INSERT INTO keyword_fts(rowid, body, title)
        SELECT row_id, body, title FROM keyword_index_content AS c
        WHERE c.row_id = (SELECT row_id FROM keyword_index_rows
                         WHERE chunk_set_id = NEW.chunk_set_id AND chunk_index = NEW.chunk_index);
    END""",
    "keyword_chunk_identity": """BEFORE UPDATE OF chunk_set_id, chunk_index, chunk_id ON chunks
        WHEN NEW.chunk_set_id IS NOT OLD.chunk_set_id OR NEW.chunk_index IS NOT OLD.chunk_index
          OR NEW.chunk_id IS NOT OLD.chunk_id
    BEGIN SELECT RAISE(ABORT, 'chunk identity is immutable'); END""",
    "keyword_set_unpublish": """AFTER UPDATE OF status ON chunk_sets
        WHEN OLD.status = 'ready' AND NEW.status <> 'ready'
    BEGIN
        DELETE FROM keyword_index_rows WHERE chunk_set_id = OLD.chunk_set_id;
        DELETE FROM keyword_index_sets WHERE chunk_set_id = OLD.chunk_set_id;
    END""",
}

CREATE_TRIGGERS = tuple(f"CREATE TRIGGER {name} {sql}" for name, sql in INDEX_TRIGGERS.items())
SCHEMA_V6 = (*INDEX_STORAGE, *CREATE_TRIGGERS)

FTS_SHADOW_TABLES = ("keyword_fts_data", "keyword_fts_idx", "keyword_fts_docsize", "keyword_fts_config")
INDEX_OBJECTS = {
    "keyword_index_rows", "keyword_index_sets", "keyword_index_content", "keyword_fts",
    *FTS_SHADOW_TABLES, *INDEX_TRIGGERS,
}
