import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
FORBIDDEN = re.compile(r"\.env\b|load_dotenv\(|dotenv_values|copytree|\bshutil\b|\bcp -[a-zA-Z]*[rR]")


# Decision 10: the only code that may load the secrets file, and only this exact line.
ALLOWED = {"train_two_tower.py": ['load_dotenv(Path(__file__).with_name(".env"))']}


def lines_that_may_copy(text, allowed=()):
    return [n for n, line in enumerate(text.splitlines(), 1) if FORBIDDEN.search(line) and line.strip() not in allowed]


class NoSecretCopies(unittest.TestCase):
    """No code here may copy a whole folder or touch the secrets file."""

    def test_pattern_catches_the_ways_to_copy(self):
        for line in ("shutil.copytree('.', x)", "cp -R . x", "cp -r . x", "cp -aR . x", "open('.env')"):
            self.assertEqual(lines_that_may_copy(line), [1])
        self.assertEqual(lines_that_may_copy("os.environ"), [])

    def test_no_file_copies_folders_or_touches_the_secrets_file(self):
        files = [f for f in ROOT.glob("**/*.py") if f.name != Path(__file__).name and ".venv" not in f.parts]
        self.assertGreater(len(files), 0)
        found = {f.name: lines_that_may_copy(f.read_text(encoding="utf-8"), ALLOWED.get(f.name, ())) for f in files}
        self.assertEqual({name: lines for name, lines in found.items() if lines}, {})

    def test_the_exception_covers_only_the_exact_line(self):
        allowed = ALLOWED["train_two_tower.py"]
        self.assertEqual(lines_that_may_copy(allowed[0], allowed), [])
        self.assertEqual(lines_that_may_copy(allowed[0]), [1])
        for line in ("print(open('.env').read())", "load_dotenv()"):
            self.assertEqual(lines_that_may_copy(line, allowed), [1])

    def test_git_ignores_the_secrets_file(self):
        self.assertIn(".env", (ROOT / ".gitignore").read_text(encoding="utf-8").split())


if __name__ == "__main__":
    unittest.main()
