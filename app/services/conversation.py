from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from app.core.config import settings


class ConversationManager:
    """세션별 채팅 대화 히스토리를 메모리에서 관리한다. 서버 재시작 시 기록은 소실된다."""

    def __init__(self):
        # 존재하지 않는 세션 키에 접근해도 KeyError 없이 빈 리스트를 반환한다.
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        대화 메시지를 세션에 추가하고, 최대 메시지 수를 초과하면 오래된 것부터 제거한다.

        Parameters:
            session_id: 대화 세션 식별자
            role      : "user" 또는 "assistant"
            content   : 메시지 내용
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversations[session_id].append(message)

        # MAX_HISTORY_MESSAGES는 user 기준이므로 user+assistant 쌍을 고려해 *2 적용
        max_messages = settings.MAX_HISTORY_MESSAGES * 2
        if len(self.conversations[session_id]) > max_messages:
            self.conversations[session_id] = self.conversations[session_id][-max_messages:]

    def get_conversation(self, session_id: str) -> List[Dict]:
        """
        세션의 전체 대화 히스토리를 반환한다.
        .get()을 사용해 defaultdict의 빈 리스트 자동 생성 부작용을 방지한다.
        """
        return self.conversations.get(session_id, [])

    def clear_conversation(self, session_id: str) -> None:
        """특정 세션의 대화 히스토리를 초기화한다."""
        if session_id in self.conversations:
            del self.conversations[session_id]


# 모든 API 요청이 이 인스턴스를 공유해야 session_id로 대화 이력을 연속적으로 추적할 수 있다.
conversation_manager = ConversationManager()
