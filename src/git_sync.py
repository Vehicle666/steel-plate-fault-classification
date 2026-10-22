"""Commit & push kết quả lên GitHub — dùng trong Colab để không mất tiến độ.

Xác thực: đặt biến môi trường `GH_TOKEN` (hoặc `GITHUB_TOKEN`) là Personal Access
Token có quyền `Contents: Read and write`. Không bao giờ commit token vào repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GitSyncError(RuntimeError):
    """Lỗi khi thao tác git (thiếu token, conflict, không có remote...)."""


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True
    )
    if check and result.returncode != 0:
        raise GitSyncError(
            f"git {' '.join(args)} thất bại:\n{result.stderr or result.stdout}"
        )
    return result


def current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def ensure_identity(name: str | None = None, email: str | None = None) -> None:
    """Đặt user.name/user.email nếu môi trường chưa có (Colab mặc định không có)."""
    name = name or os.getenv("GIT_AUTHOR_NAME") or "colab-runner"
    email = email or os.getenv("GIT_AUTHOR_EMAIL") or "colab-runner@users.noreply.github.com"
    if not _git("config", "--get", "user.name", check=False).stdout.strip():
        _git("config", "user.name", name)
    if not _git("config", "--get", "user.email", check=False).stdout.strip():
        _git("config", "user.email", email)


def origin_url() -> str:
    return _git("remote", "get-url", "origin").stdout.strip()


def _url_with_token(url: str, token: str) -> str:
    """Nhúng token vào URL https, bỏ credential cũ nếu có."""
    if not url.startswith("https://"):
        return url
    rest = url[len("https://") :]
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    return f"https://x-access-token:{token}@{rest}"


def has_changes(paths: list[str] | None = None) -> bool:
    _git("add", "-A", "--", *(paths or ["."]))
    return bool(_git("status", "--porcelain").stdout.strip())


def commit_and_push(
    message: str,
    paths: list[str] | None = None,
    push: bool = True,
    branch: str | None = None,
    retries: int = 3,
) -> bool:
    """Commit các thay đổi và push lên GitHub.

    Trả về True nếu có commit mới, False nếu không có gì thay đổi.
    """
    ensure_identity()

    if not has_changes(paths):
        return False

    _git("commit", "-m", message)
    print(f"  [git] đã commit: {message}")

    if not push:
        return True

    branch = branch or current_branch()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    original = origin_url()

    if token:
        _git("remote", "set-url", "origin", _url_with_token(original, token))
    elif "x-access-token" not in original and "@" not in original:
        raise GitSyncError(
            "Chưa có GH_TOKEN. Trên Colab: thêm secret tên GH_TOKEN "
            "(Personal Access Token, quyền Contents: Read and write) rồi chạy lại."
        )

    try:
        for attempt in range(1, retries + 1):
            pull = _git("pull", "--rebase", "--autostash", "origin", branch, check=False)
            if pull.returncode != 0:
                _git("rebase", "--abort", check=False)

            pushed = _git("push", "origin", f"HEAD:{branch}", check=False)
            if pushed.returncode == 0:
                print(f"  [git] đã push lên origin/{branch}")
                return True

            print(f"  [git] push lần {attempt} thất bại, thử lại...")

        # Runtime chỉ có một writer (chính nó) nên force-with-lease an toàn ở đây.
        forced = _git("push", "--force-with-lease", "origin", f"HEAD:{branch}", check=False)
        if forced.returncode == 0:
            print(f"  [git] đã push (force-with-lease) lên origin/{branch}")
            return True
        raise GitSyncError(
            "Push thất bại. Kiểm tra GH_TOKEN còn hạn và có quyền Contents: write."
        )
    finally:
        if token:
            _git("remote", "set-url", "origin", original)
