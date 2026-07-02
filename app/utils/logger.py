import logging
import json
from datetime import datetime


def setup_logger(name: str) -> logging.Logger:
    """
    애플리케이션 로거를 생성하고 반환한다.

    Parameters:
        name: 로거 식별자. 동일한 name을 여러 모듈에서 사용하면 같은 인스턴스를 공유한다.

    Returns:
        콘솔 출력 핸들러가 붙은 logging.Logger 인스턴스
    """
    logger = logging.getLogger(name)
    # 기본값(WARNING)을 재정의해 DEBUG/INFO 로그도 출력되게 한다.
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger("RAGServer")
