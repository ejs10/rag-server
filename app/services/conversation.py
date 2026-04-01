from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from app.core.config import settings
from app.utils.logger import logger

class ConversationManager:
    """ 대화 관리 서비스 (대화 히스토리 저장 및 검색) """

    def __init__(self):
        # session_id -> 메시지 리스트
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """대화 추가"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversations[session_id].append(message)

        # 최대 메시지 수 초과 시 오래된 것부터 제거
        max_messages = settings.MAX_HISTORY_MESSAGES *2
        if len(self.conversations[session_id]) > max_messages:
            self.conversations[session_id] = self.conversations[session_id][-max_messages:]


    def get_conversation(self, session_id: str) -> List[Dict]:
        """특정 세션의 대화 히스토리 조회"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str) -> None:
        """특정 세션의 대화 히스토리 초기화"""
        if session_id in self.conversations:
            del self.conversations[session_id]

conversation_manager = ConversationManager()