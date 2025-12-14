from lexi.core.config import LexiConfig
from lexi.core.constants import LexiMode
from lexi.chrono.adapter import ChronoAdapter
from lexi.lex.nano.pipeline import LexiNanoPipeline
from lexi.lex.lds.pipeline import LexiLDSPipeline
from lexi.core.constants import FileStatus

class LexiController:
    """
       Global orchestrator for Chrono ↔ Lex sync
       """

    def __init__(self, lexi_configuration: dict):
        self.config = LexiConfig(lexi_configuration)

        self._init_chrono()
        self._init_lex()

    def _init_chrono(self):
        chrono_config = {
            "supported_file_types": self._get_supported_file_types(),
            "db_path": self.config.db_path,
            "scan_dirs": self.config.sync_dirs,
            "in_memory": self.config.in_memory,
            "use_hash_for_changes": self.config.use_hash_for_changes
        }

        self.chrono = ChronoAdapter(chrono_config)

    def _handle_new(self, record: dict):
        self.lex.ingest(record)

    def _handle_changed(self, record: dict):
        file_path = record["file_path"]
        self.lex.delete(file_path)
        self.lex.ingest(record)

    def _handle_deleted(self, file_path: str):
        self.lex.delete(file_path)

    def _init_lex(self):
        if self.config.mode == LexiMode.LEXI_NANO:
            self.lex = LexiNanoPipeline(self.config)
        else:
            self.lex = LexiLDSPipeline(self.config)



    def search(self, query: str, **kwargs):
        return self.lex.search(query, **kwargs)

    def shutdown(self):
        self.chrono.close()

    def _get_supported_file_types(self):
        if self.config.mode.value == "LEXI_NANO":
            return [".txt", ".pdf", ".doc", ".docx"]

        # LEXI_LDS (future ready)
        return [
            ".txt", ".pdf", ".doc", ".docx", ".xlsx",
            ".json", ".sql", ".png", ".jpg", ".jpeg",
            ".mp3", ".mp4", ".wav"
        ]

    def sync(self):
        changes = self.chrono.scan()
        for record in self.chrono.get_active_records():
            if record["status"] == FileStatus.STATUS_NEW:
                self.lex.ingest(record)
                self.chrono.update_status(record["rowid"], FileStatus.STATUS_SYNCED)
            elif record["status"] == FileStatus.STATUS_CHANGED:
                self.lex.delete(record["path"])
                self.lex.ingest(record)
                self.chrono.update_status(record["rowid"], FileStatus.STATUS_SYNCED)
            elif record["status"] == FileStatus.STATUS_DELETED:
                self.lex.delete(record["path"])
                self.chrono.remove_record(record["rowid"])

        self.chrono.commit()

        # 5️⃣ Save FAISS index to disk if configured
        if not getattr(self.config, "faiss_in_memory", False):
            try:
                self.lex.persist()
                print("[Lexi] FAISS index persisted to disk.")
            except Exception as e:
                print(f"[Lexi] Warning: failed to persist FAISS index → {e}")



    def stats(self):
        return {
            "chunks": self.lex.metastore.stats(),
            "mode": self.config.mode.value,
            "directories": len(self.config.sync_dirs)
        }

    def persist(self):
        self.faiss_store.save(self.index_path)

    def load(self):
        self.faiss_store.load(self.index_path)

    def shutdown(self):
        self.lex.persist()
        self.chrono.close()








