import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가해 `app.*` 패키지를 어떤 방식으로 pytest를 실행해도 임포트할 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
