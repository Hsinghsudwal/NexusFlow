import subprocess
import uuid
from typing import Optional

class PipelineVersioner:
    def __init__(self):
        self.run_id = str(uuid.uuid4())
        self.git_commit = self._get_git_commit()

    def _get_git_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
        except Exception:
            return None

    def get_version_info(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "version_date": datetime.now().isoformat()
        }