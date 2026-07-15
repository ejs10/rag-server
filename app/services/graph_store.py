import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from app.core.config import settings
from app.utils.logger import logger
from app.services.llm import get_llm, apply_llm_retry
from langchain_core.prompts import PromptTemplate


class Node(BaseModel):
    """지식 그래프의 엔티티 노드."""
    id: str = Field(description="고유 엔티티 이름 (예: 사람, 조직, 기술, 개념 등)")
    type: str = Field(description="엔티티 유형 (대문자 영문, 예: PERSON, ORG, CONCEPT, TECHNOLOGY)")


class Relationship(BaseModel):
    """노드와 노드 사이의 방향이 있는 관계."""
    source: str = Field(description="출발 엔티티 이름 (Node.id와 정확히 일치)")
    target: str = Field(description="도착 엔티티 이름 (Node.id와 정확히 일치)")
    type: str = Field(description="관계 유형 (대문자 영문, 예: FOUNDED, USES, RELATED_TO)")
    description: str = Field(description="관계에 대한 구체적인 설명")


class KnowledgeGraph(BaseModel):
    """텍스트에서 추출한 지식 그래프 (노드 목록 + 관계 목록)."""
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class GraphStore:
    """Neo4j를 이용한 지식 그래프 저장 및 검색 서비스."""

    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self.driver = None
        self._connect()

    def _connect(self):
        """Neo4j 서버에 연결한다. 실패해도 서버 전체가 중단되지 않도록 에러를 잡는다."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Neo4j 그래프 DB 연결 완료")
            self._init_schema()
        except Exception as e:
            # self.driver=None이면 이후 메서드들이 None 체크 후 조용히 스킵한다.
            logger.error(f"Neo4j 연결 실패: {e}")

    def _init_schema(self):
        """엔티티 ID 유일성 제약조건을 생성한다. 없으면 같은 엔티티가 중복 추가될 수 있다."""
        if not self.driver:
            return
        query = "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
                logger.debug("Neo4j 스키마 제약조건 확인/생성 완료")
        except Exception as e:
            logger.warning(f"Neo4j 스키마 생성 오류: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def extract_graph_from_text(self, text: str) -> KnowledgeGraph:
        """
        LLM을 사용해 텍스트에서 엔티티와 관계를 자동 추출한다.
        with_structured_output(KnowledgeGraph)으로 LLM이 JSON 스키마에 맞게 응답하도록 강제한다.

        Parameters:
            text: 분석할 텍스트

        Returns:
            추출된 KnowledgeGraph 객체. 실패 시 빈 그래프를 반환한다.
        """
        provider = settings.GRAPH_EXTRACTION_MODEL if settings.GRAPH_EXTRACTION_MODEL else settings.LLM_PROVIDER
        try:
            # RunnableRetry(재시도 래퍼)는 BaseChatModel의 with_structured_output을 위임하지 않는다.
            # 원본 모델에 structured output을 먼저 적용한 뒤, 완성된 파이프라인에 재시도를 씌운다.
            llm = get_llm(provider, temperature=0.1, use_retry=False)
            structured_llm = apply_llm_retry(llm.with_structured_output(KnowledgeGraph))

            prompt = f"""다음 텍스트를 분석하여 중요한 지식 그래프 엔티티와 관계를 추출하세요.
엔티티는 구체적인 명사(사람, 조직, 기술, 장소, 핵심 개념)로 한정하세요.
관계(Relationship)의 source와 target은 반드시 추출한 엔티티의 id와 정확히 일치해야 합니다.

텍스트:
{text}
"""
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"지식 그래프 추출 실패: {e}")
            return KnowledgeGraph()

    def save_graph(self, kg: KnowledgeGraph, document_id: str):
        """
        추출된 KnowledgeGraph를 Neo4j에 저장한다.
        MERGE를 사용하므로 여러 문서에 같은 엔티티가 등장해도 중복 없이 합쳐진다.

        Parameters:
            kg         : 저장할 지식 그래프
            document_id: 출처 문서 ID (관계의 document_id 속성으로 저장)
        """
        if not self.driver or not kg.nodes:
            return

        try:
            with self.driver.session(database=self.database) as session:
                for node in kg.nodes:
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        SET n.type = $type
                        """,
                        id=node.id, type=node.type
                    )
                for rel in kg.relationships:
                    session.run(
                        """
                        MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
                        MERGE (a)-[r:RELATION {type: $type}]->(b)
                        SET r.description = $desc, r.document_id = $doc_id
                        """,
                        source=rel.source, target=rel.target,
                        type=rel.type, desc=rel.description, doc_id=document_id
                    )
            logger.debug(f"문서 {document_id}의 그래프 데이터 저장 완료: 노드 {len(kg.nodes)}개, 관계 {len(kg.relationships)}개")
        except Exception as e:
            logger.error(f"Neo4j 그래프 저장 실패: {e}")

    def delete_graph(self, document_id: str) -> None:
        """
        특정 문서와 연결된 관계를 Neo4j에서 삭제하고, 연결이 끊긴 고아 노드를 정리한다.

        Parameters:
            document_id: 삭제할 문서의 고유 ID
        """
        if not self.driver:
            return
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MATCH ()-[r:RELATION {document_id: $doc_id}]->() DELETE r",
                    doc_id=document_id
                )
                # 모든 관계가 제거된 고아 노드는 이후 검색에서 참조될 수 없으므로 함께 정리한다.
                session.run(
                    "MATCH (n:Entity) WHERE NOT (n)-[:RELATION]-() DELETE n"
                )
            logger.info(f"그래프 데이터 삭제 완료: {document_id}")
        except Exception as e:
            logger.error(f"그래프 데이터 삭제 실패: {e}")

    def search_graph(self, query: str) -> List[Dict]:
        """
        질문에서 핵심 키워드를 추출하고, 해당 엔티티의 관계를 Neo4j에서 조회한다.

        Parameters:
            query: 검색 질문 문자열

        Returns:
            중복 제거된 관계 딕셔너리 목록 (source, type, target, description)
        """
        if not self.driver:
            return []

        try:
            llm = get_llm(temperature=0.0)
            keyword_prompt = f"다음 질문에서 가장 핵심적인 엔티티(명사) 키워드 3개만 쉼표로 구분하여 출력하세요. 설명 없이 키워드만 나열하세요.\n질문: {query}"
            keywords_str = llm.invoke(keyword_prompt).text
            # 빈 키워드를 걸러내지 않으면 Cypher의 CONTAINS ''가 모든 엔티티에 매칭되어
            # 질문과 무관한 관계가 결과에 섞여 들어간다.
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            if not keywords:
                logger.debug("그래프 검색: 유효한 키워드를 추출하지 못해 검색을 건너뜁니다")
                return []

            results = []
            with self.driver.session(database=self.database) as session:
                for kw in keywords:
                    # CONTAINS: 부분 문자열 매칭으로 "구글" 키워드로 "구글코리아"도 검색한다.
                    cypher = """
                    MATCH (n:Entity)-[r:RELATION]->(m:Entity)
                    WHERE n.id CONTAINS $kw OR m.id CONTAINS $kw
                    RETURN n.id AS source, r.type AS rel_type, r.description AS desc, m.id AS target
                    LIMIT 5
                    """
                    records = session.run(cypher, kw=kw)
                    for record in records:
                        results.append({
                            "source": record["source"],
                            "type": record["rel_type"],
                            "target": record["target"],
                            "description": record["desc"]
                        })

            # 여러 키워드가 같은 관계를 반환할 수 있으므로 서명(source-type-target)으로 중복 제거한다.
            unique_results = []
            seen = set()
            for r in results:
                sig = f"{r['source']}-{r['type']}-{r['target']}"
                if sig not in seen:
                    seen.add(sig)
                    unique_results.append(r)

            logger.debug(f"그래프 검색 결과: {len(unique_results)}개 관계 찾음 (키워드: {keywords})")
            return unique_results

        except Exception as e:
            logger.error(f"그래프 검색 실패: {e}")
            return []
