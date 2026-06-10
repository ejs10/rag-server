import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from app.core.config import settings
from app.utils.logger import logger
from app.services.llm import get_llm
from langchain_core.prompts import PromptTemplate

class Node(BaseModel):
    id: str = Field(description="고유 엔티티 이름 (예: 사람, 조직, 기술, 개념 등)")
    type: str = Field(description="엔티티 유형 (대문자 영문, 예: PERSON, ORG, CONCEPT, TECHNOLOGY)")

class Relationship(BaseModel):
    source: str = Field(description="출발 엔티티 이름 (Node.id와 정확히 일치)")
    target: str = Field(description="도착 엔티티 이름 (Node.id와 정확히 일치)")
    type: str = Field(description="관계 유형 (대문자 영문, 예: FOUNDED, USES, RELATED_TO)")
    description: str = Field(description="관계에 대한 구체적인 설명")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

class GraphStore:
    """Neo4j를 이용한 지식 그래프 저장 및 검색 서비스"""
    
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self.driver = None
        self._connect()
        
    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Neo4j 그래프 DB 연결 완료")
            self._init_schema()
        except Exception as e:
            logger.error(f"Neo4j 연결 실패: {e}")
            
    def _init_schema(self):
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
        """LLM을 사용하여 텍스트에서 엔티티와 관계를 추출합니다."""
        provider = settings.GRAPH_EXTRACTION_MODEL if settings.GRAPH_EXTRACTION_MODEL else settings.LLM_PROVIDER
        try:
            llm = get_llm(provider, temperature=0.1)
            structured_llm = llm.with_structured_output(KnowledgeGraph)
            
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
        """추출된 KnowledgeGraph 객체를 Neo4j에 저장합니다."""
        if not self.driver or not kg.nodes:
            return
            
        try:
            with self.driver.session(database=self.database) as session:
                # 노드 생성
                for node in kg.nodes:
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        SET n.type = $type
                        """,
                        id=node.id, type=node.type
                    )
                # 관계 생성
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

    def search_graph(self, query: str) -> List[Dict]:
        """질문과 관련된 엔티티를 찾고, 해당 엔티티들의 관계 정보를 조회합니다."""
        if not self.driver:
            return []
            
        # 1. 쿼리에서 주요 엔티티 키워드 추출 (간단히 명사 위주 또는 LLM을 다시 사용)
        # 비용/시간 문제를 줄이기 위해 간단히 LLM으로 키워드만 뽑음
        try:
            llm = get_llm(temperature=0.0)
            keyword_prompt = f"다음 질문에서 가장 핵심적인 엔티티(명사) 키워드 3개만 쉼표로 구분하여 출력하세요. 설명 없이 키워드만 나열하세요.\n질문: {query}"
            keywords_str = llm.invoke(keyword_prompt).content
            keywords = [k.strip() for k in keywords_str.split(',')]
            
            results = []
            with self.driver.session(database=self.database) as session:
                for kw in keywords:
                    # 해당 키워드를 포함하는 노드의 1-hop 관계 조회
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
            
            # 중복 제거
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
