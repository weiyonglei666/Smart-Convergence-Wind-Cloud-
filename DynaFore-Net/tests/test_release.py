import pathlib
import subprocess
import sys
import tempfile
import unittest


class ReleaseTests(unittest.TestCase):
    def test_audit_rejects_private_arrays_and_symlinks(self):
        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
        self.assertTrue(script.exists(), "Release audit script not created yet")
        with tempfile.TemporaryDirectory(prefix="dynafore_release_") as tmp:
            root = pathlib.Path(tmp)
            (root / "README.md").write_text("Contact ts26170007a31@cumt.edu.cn", encoding="utf-8")
            for name in ["README_EN.md", "DATA_AVAILABILITY.md"]:
                (root / name).write_text("Contact ts26170007a31@cumt.edu.cn", encoding="utf-8")
            command = [sys.executable, str(script), "--root", str(root)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            private = root / "private.npy"
            private.write_bytes(b"runtime test fixture only")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("private.npy", result.stdout + result.stderr)
            private.unlink()
            secret = root / "key.txt"
            secret.write_text("-----" + "BEGIN PRIVATE KEY" + "-----\nnot-a-real-key\n", encoding="utf-8")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            secret.unlink()
            (root / "linked.md").symlink_to(root / "README.md")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
