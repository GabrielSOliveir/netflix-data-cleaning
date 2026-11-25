import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
FORBIDDEN = re.compile(r"\.env\b|copytree|\bshutil\b|\bcp -[a-zA-Z]*[rR]")


def lines_that_may_copy(text):
    return [n for n, line in enumerate(text.splitlines(), 1) if FORBIDDEN.search(line)]


class NoSecretCopies(unittest.TestCase):
    """No code here may copy a whole folder or touch the secrets file."""

    def test_pattern_catches_the_ways_to_copy(self):
        for line in ("shutil.copytree('.', x)", "cp -R . x", "cp -r . x", "cp -aR . x", "open('.env')"):
            self.assertEqual(lines_that_may_copy(line), [1])
        self.assertEqual(lines_that_may_copy("os.environ"), [])

    def test_no_file_copies_folders_or_touches_the_secrets_file(self):
        files = [f for f in ROOT.glob("**/*.py") if f.name != Path(__file__).name and ".venv" not in f.parts]
        self.assertGreater(len(files), 0)
        self.assertEqual({f.name: lines_that_may_copy(f.read_text(encoding="utf-8")) for f in files
                          if lines_that_may_copy(f.read_text(encoding="utf-8"))}, {})


if __name__ == "__main__":
    unittest.main()
