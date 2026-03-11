#!/usr/bin/env python3
"""
Agent Memory System - 每个 Agent 独立的记忆系统

Architecture:
- 每个 Agent 有独立的记忆库、知识图谱、工作记忆
- 通过 agent_id 隔离
- 智能压缩：保留关键信息 50-60%

Memory Compression:
- Filter: 打招呼、应答词、思考过程
- Keep: 决策、事件、偏好、事实、上下文
"""

import os
import json
import re
import requests
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class CompressionConfig:
    """记忆压缩配置"""
    # 过滤模式（不存储）
    filter_patterns = [
        r'^(你好|嗨|hello|hi|hey|早上好|晚上好)[！!。.]*$',
        r'^(好的|明白了|收到|OK|ok|嗯|哦|啊)[！!。.]*$',
        r'^(谢谢|感谢|多谢)[！!。.]*$',
        r'^(让我想想|嗯\.\.\.|啊\.\.\.|思考中)',
        r'^(没问题|不客气|不用谢)',
    ]
    
    # 保留关键词（高优先级）
    keep_keywords = [
        '决定', '完成', '创建', '配置', '安装', '部署', '修复', '更新',
        '记住', '记住这个', '别忘了', '重要',
        '偏好', '喜欢', '习惯',
        '密码', '密钥', 'token', 'key', 'secret',
        '问题是', '原因是', '解决方案',
    ]
    
    # 压缩比例
    min_keep_ratio = 0.5   # 最少保留 50%
    max_keep_ratio = 0.6   # 最多保留 60%


class MemoryCompressor:
    """记忆压缩器"""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
    
    def should_filter(self, text: str) -> bool:
        """判断是否应该过滤"""
        text = text.strip().lower()
        
        for pattern in self.config.filter_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_importance(self, text: str) -> float:
        """计算重要性分数 (0-1)"""
        score = 0.5  # 基础分
        
        # 关键词加分
        for kw in self.config.keep_keywords:
            if kw.lower() in text.lower():
                score += 0.15
        
        # 包含具体信息（路径、URL、数字）
        if re.search(r'[/~]\w+', text):  # 路径
            score += 0.1
        if re.search(r'https?://', text):  # URL
            score += 0.1
        if re.search(r'\d{4,}', text):  # 长数字（可能是配置）
            score += 0.05
        
        # 长度奖励（内容丰富）
        if len(text) > 100:
            score += 0.1
        elif len(text) > 50:
            score += 0.05
        
        return min(1.0, score)
    
    def compress_conversation(self, messages: List[str]) -> List[Tuple[str, float]]:
        """
        压缩对话
        
        Args:
            messages: 原始消息列表
        
        Returns:
            [(compressed_text, importance), ...]
        """
        results = []
        
        for msg in messages:
            # 过滤无意义内容
            if self.should_filter(msg):
                continue
            
            # 计算重要性
            importance = self.get_importance(msg)
            
            # 只保留重要性 >= 0.5 的内容
            if importance >= 0.5:
                # 压缩：移除多余空白
                compressed = ' '.join(msg.split())
                results.append((compressed, importance))
        
        # 按重要性排序，保留 top 50-60%
        if len(results) > 10:
            results.sort(key=lambda x: x[1], reverse=True)
            keep_count = int(len(results) * self.config.max_keep_ratio)
            results = results[:keep_count]
        
        return results


class AgentMemoryStore:
    """
    Agent 记忆存储
    
    每个 Agent 独立：
    - 向量记忆库 (Weaviate)
    - 按 agent_id 隔离
    """
    
    def __init__(self, agent_id: str, user_id: str = "default",
                 weaviate_url: str = "http://localhost:8080",
                 ollama_url: str = "http://localhost:11434"):
        self.agent_id = agent_id
        self.user_id = user_id
        self.weaviate_url = weaviate_url
        self.ollama_url = ollama_url
        self.embedding_model = "nomic-embed-text"
        
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取向量嵌入"""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.embedding_model, "input": text},
            proxies={"http": None, "https": None}
        )
        return resp.json()["embeddings"][0]
    
    def add_memory(self, content: str, memory_type: str = "context",
                   importance: float = 0.5, tags: List[str] = None) -> str:
        """
        添加记忆到 Agent 的记忆库
        
        按 agent_id 隔离
        """
        obj = {
            "class": "Memory",
            "properties": {
                "content": content,
                "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "type": memory_type,
                "importance": importance,
                "tags": tags or [],
                "agent_id": self.agent_id,    # Agent 隔离
                "user_id": self.user_id        # 用户归属（但 Agent 独立）
            }
        }
        
        resp = requests.post(
            f"{self.weaviate_url}/v1/objects",
            json=obj,
            proxies={"http": None, "https": None}
        )
        
        if resp.status_code == 200:
            return resp.json().get("id", "")
        raise Exception(f"添加记忆失败: {resp.text}")
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        搜索 Agent 的记忆
        
        只返回当前 Agent 的记忆
        """
        query_vector = self._get_embedding(query)
        
        # 只查当前 Agent
        conditions = [{
            "path": ["agent_id"],
            "operator": "Equal",
            "valueText": self.agent_id
        }]
        
        graphql_query = {
            "query": f'''{{
                Get {{
                    Memory(
                        nearVector: {{vector: {json.dumps(query_vector)}}}
                        where: {{operator: "And", operands: {json.dumps(conditions)}}}
                        limit: {limit}
                    ) {{
                        _additional {{ id certainty }}
                        content date type importance tags
                    }}
                }}
            }}'''
        }
        
        resp = requests.post(
            f"{self.weaviate_url}/v1/graphql",
            json=graphql_query,
            proxies={"http": None, "https": None}
        )
        
        return resp.json().get("data", {}).get("Get", {}).get("Memory", [])
    
    def get_memories_by_date(self, date: str) -> List[Dict]:
        """获取指定日期的记忆"""
        conditions = [
            {"path": ["agent_id"], "operator": "Equal", "valueText": self.agent_id},
            {"path": ["date"], "operator": "Like", "valueText": f"{date}T*"}
        ]
        
        graphql_query = {
            "query": f'''{{
                Get {{
                    Memory(
                        where: {{operator: "And", operands: {json.dumps(conditions)}}}
                        limit: 100
                    ) {{
                        _additional {{ id }}
                        content date type importance tags
                    }}
                }}
            }}'''
        }
        
        resp = requests.post(
            f"{self.weaviate_url}/v1/graphql",
            json=graphql_query,
            proxies={"http": None, "https": None}
        )
        
        return resp.json().get("data", {}).get("Get", {}).get("Memory", [])
    
    def get_date_list(self) -> Dict[str, int]:
        """获取所有有记忆的日期"""
        conditions = [{
            "path": ["agent_id"],
            "operator": "Equal",
            "valueText": self.agent_id
        }]
        
        graphql_query = {
            "query": f'''{{
                Get {{
                    Memory(
                        where: {{operator: "And", operands: {json.dumps(conditions)}}}
                        limit: 1000
                    ) {{
                        date
                    }}
                }}
            }}'''
        }
        
        resp = requests.post(
            f"{self.weaviate_url}/v1/graphql",
            json=graphql_query,
            proxies={"http": None, "https": None}
        )
        
        memories = resp.json().get("data", {}).get("Get", {}).get("Memory", [])
        
        date_counts = {}
        for m in memories:
            date = (m.get("date", "")[:10])
            date_counts[date] = date_counts.get(date, 0) + 1
        
        return date_counts


class AgentKnowledgeGraph:
    """
    Agent 知识图谱
    
    每个 Agent 独立，存储在 SQLite
    """
    
    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.user_id = user_id
        
        # 每个 Agent 有独立的数据库文件
        db_dir = os.path.expanduser(f"~/.openclaw/users/{user_id}/agents/{agent_id}")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "knowledge_graph.db")
        
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
    
    def _init_db(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                entity_type TEXT,
                description TEXT,
                mention_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT
            );
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                weight REAL,
                evidence TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
        ''')
        self.conn.commit()
    
    def add_entity(self, name: str, entity_type: str, description: str = ""):
        """添加实体"""
        import uuid
        now = datetime.now().isoformat()
        
        cursor = self.conn.execute(
            "SELECT id, mention_count FROM entities WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        
        if row:
            self.conn.execute(
                "UPDATE entities SET mention_count = ?, last_seen = ? WHERE id = ?",
                (row[1] + 1, now, row[0])
            )
        else:
            self.conn.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4())[:8], name, entity_type, description, 1, now, now)
            )
        
        self.conn.commit()
    
    def get_entities(self, limit: int = 100) -> List[Dict]:
        """获取所有实体"""
        entities = []
        for row in self.conn.execute(
            "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
            (limit,)
        ):
            entities.append({
                "id": row[0],
                "name": row[1],
                "entityType": row[2],
                "description": row[3],
                "mentionCount": row[4]
            })
        return entities
    
    def add_relation(self, source: str, target: str, relation_type: str, evidence: str = ""):
        """添加关系"""
        import uuid
        
        # 获取实体 ID
        source_cursor = self.conn.execute(
            "SELECT id FROM entities WHERE name = ?", (source,)
        )
        target_cursor = self.conn.execute(
            "SELECT id FROM entities WHERE name = ?", (target,)
        )
        
        source_row = source_cursor.fetchone()
        target_row = target_cursor.fetchone()
        
        if source_row and target_row:
            self.conn.execute(
                "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4())[:8], source_row[0], target_row[0], 
                 relation_type, 1.0, evidence)
            )
            self.conn.commit()


class AgentWorkingMemory:
    """
    Agent 工作记忆
    
    会话级存储，任务后清理
    """
    
    def __init__(self, agent_id: str, max_items: int = 20):
        self.agent_id = agent_id
        self.max_items = max_items
        self.memories: List[Dict] = []
    
    def add(self, content: str, memory_type: str = "context"):
        """添加到工作记忆"""
        self.memories.append({
            "content": content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.memories) > self.max_items:
            self.memories.pop(0)
    
    def get_context(self, max_tokens: int = 2000) -> str:
        """获取上下文"""
        result = ""
        for m in reversed(self.memories):
            if len(result) + len(m["content"]) > max_tokens:
                break
            result = m["content"] + "\n" + result
        return result
    
    def clear(self):
        """清理工作记忆"""
        self.memories = []


class AgentMemorySystem:
    """
    Agent 记忆系统 - 统一入口
    
    每个 Agent 独立：
    - AgentMemoryStore (Weaviate)
    - AgentKnowledgeGraph (SQLite)
    - AgentWorkingMemory (内存)
    
    记忆压缩：
    - 过滤无用内容
    - 保留 50-60% 关键信息
    """
    
    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.user_id = user_id
        
        # Agent 独立的存储
        self.memory_store = AgentMemoryStore(agent_id, user_id)
        self.knowledge_graph = AgentKnowledgeGraph(agent_id, user_id)
        self.working_memory = AgentWorkingMemory(agent_id)
        
        # 压缩器
        self.compressor = MemoryCompressor()
    
    def remember(self, content: str, memory_type: str = "context",
                 importance: float = None) -> str:
        """
        存储记忆（自动压缩）
        
        Args:
            content: 记忆内容
            memory_type: 类型
            importance: 重要性（自动计算）
        
        Returns:
            记忆 ID
        """
        # 检查是否应该过滤
        if self.compressor.should_filter(content):
            return None
        
        # 计算重要性
        if importance is None:
            importance = self.compressor.get_importance(content)
        
        # 只存储重要性 >= 0.5 的内容
        if importance < 0.5:
            return None
        
        # 存到长期记忆
        mem_id = self.memory_store.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance
        )
        
        # 存到工作记忆
        self.working_memory.add(content, memory_type)
        
        # 提取实体到知识图谱
        self._extract_entities(content)
        
        return mem_id
    
    def remember_conversation(self, messages: List[str]) -> List[str]:
        """
        存储对话（批量压缩）
        
        保留 50-60% 关键信息
        """
        compressed = self.compressor.compress_conversation(messages)
        
        mem_ids = []
        for text, importance in compressed:
            mem_id = self.remember(text, "context", importance)
            if mem_id:
                mem_ids.append(mem_id)
        
        return mem_ids
    
    def recall(self, query: str, limit: int = 10) -> List[Dict]:
        """
        检索记忆
        
        优先级：
        1. 工作记忆
        2. 长期记忆（语义搜索）
        """
        results = []
        
        # 工作记忆
        for m in self.working_memory.memories:
            if query.lower() in m["content"].lower():
                results.append({
                    "content": m["content"],
                    "type": m["type"],
                    "source": "working",
                    "priority": 1.0
                })
        
        # 长期记忆
        long_term = self.memory_store.search_memories(query, limit)
        for m in long_term:
            results.append({
                "content": m.get("content"),
                "type": m.get("type"),
                "date": m.get("date"),
                "source": "long_term",
                "priority": 0.8,
                "certainty": m.get("_additional", {}).get("certainty", 0)
            })
        
        return results[:limit]
    
    def recall_by_date(self, date: str) -> List[Dict]:
        """按日期检索"""
        return self.memory_store.get_memories_by_date(date)
    
    def get_date_list(self) -> Dict[str, int]:
        """获取日期列表"""
        return self.memory_store.get_date_list()
    
    def _extract_entities(self, text: str):
        """提取实体到知识图谱"""
        patterns = {
            "tool": r'\b(Docker|Python|Weaviate|Ollama|OpenClaw|Git|GitHub|Flask|Redis)\b',
            "project": r'(\w+(?:系统|模块|项目|服务))',
            "concept": r'\b(API|LLM|RAG|向量数据库|知识图谱)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for name in matches:
                if isinstance(name, tuple):
                    name = name[0]
                self.knowledge_graph.add_entity(name, entity_type)
    
    def get_context(self) -> str:
        """获取当前上下文"""
        return self.working_memory.get_context()
    
    def clear_session(self):
        """清理会话（任务完成后调用）"""
        self.working_memory.clear()
    
    def get_stats(self) -> Dict:
        """获取统计"""
        dates = self.get_date_list()
        entities = self.knowledge_graph.get_entities()
        
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "total_memories": sum(dates.values()),
            "total_dates": len(dates),
            "total_entities": len(entities),
            "working_memory_count": len(self.working_memory.memories)
        }


# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Memory System")
    parser.add_argument("--agent", default="main", help="Agent ID")
    parser.add_argument("--user", default="default", help="User ID")
    parser.add_argument("--remember", help="Store memory")
    parser.add_argument("--recall", help="Search memory")
    parser.add_argument("--dates", action="store_true", help="List dates")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--compress", help="Compress text and show result")
    
    args = parser.parse_args()
    
    memory = AgentMemorySystem(agent_id=args.agent, user_id=args.user)
    
    if args.remember:
        mem_id = memory.remember(args.remember)
        if mem_id:
            print(f"✅ 已存储: {mem_id}")
        else:
            print("⚠️ 内容被过滤（不重要）")
    
    elif args.recall:
        results = memory.recall(args.recall)
        print(f"🔍 找到 {len(results)} 条记忆:")
        for r in results:
            print(f"  [{r['source']}] {r['content'][:50]}...")
    
    elif args.dates:
        dates = memory.get_date_list()
        print(f"📅 Agent {args.agent} 的记忆日期:")
        for date, count in sorted(dates.items(), reverse=True)[:10]:
            print(f"  {date}: {count} 条")
    
    elif args.stats:
        stats = memory.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.compress:
        compressor = MemoryCompressor()
        messages = args.compress.split("|")
        compressed = compressor.compress_conversation(messages)
        print(f"📝 压缩结果 (保留 {len(compressed)}/{len(messages)}):")
        for text, imp in compressed:
            print(f"  [{imp:.2f}] {text[:50]}...")
    
    else:
        stats = memory.get_stats()
        print(f"🧠 Agent {args.agent} (User: {args.user})")
        print(f"   记忆: {stats['total_memories']}")
        print(f"   日期: {stats['total_dates']}")
        print(f"   实体: {stats['total_entities']}")


if __name__ == "__main__":
    main()