import pytest

from app.utils.file_handler import sanitize_filename


def test_sanitize_filename_keeps_normal_name():
    assert sanitize_filename("report.pdf") == "report.pdf"


def test_sanitize_filename_strips_unix_path_traversal():
    assert sanitize_filename("../../../../etc/passwd") == "passwd"


def test_sanitize_filename_strips_windows_path_traversal():
    assert sanitize_filename("..\\..\\windows\\evil.txt") == "evil.txt"


def test_sanitize_filename_strips_absolute_path():
    assert sanitize_filename("/etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Windows\\System32\\evil.txt") == "evil.txt"


@pytest.mark.parametrize("bad_name", ["", ".", "..", "../", "..\\", None])
def test_sanitize_filename_rejects_empty_or_dot_only(bad_name):
    # file.filename은 클라이언트가 파일명을 아예 지정하지 않으면 None일 수 있다.
    # None을 str 메서드로 처리하려 하면 AttributeError가 나서 upload.py의 ValueError
    # 핸들링을 빠져나가 500으로 잘못 응답하게 되므로, ValueError로 통일되어야 한다.
    with pytest.raises(ValueError):
        sanitize_filename(bad_name)
