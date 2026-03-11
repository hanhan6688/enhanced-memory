#!/usr/bin/env python3
"""
记忆系统 Web API
提供 REST API 和可视化界面
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file

# 添加父目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from api.memory_api import get_memory_system

# UI 文件路径
UI_FILE = os.path.join(BASE_DIR, 'ui', 'index.html')

app = Flask(__name__)

# ===== UI 路由 =====

@app.route('/')
def index():
    return send_file(UI_FILE)

# ===== API 路由 =====

@app.route('/api/memories', methods=['GET'])
def list_memories():
    """获取所有记忆（按 Agent 隔离）"""
    ms = get_memory_system()
    limit = request.args.get('limit', 100, type=int)
    agent_id = request.args.get('agent', 'default')
    memories = ms.get_all_memories(limit)
    
    # 按 Agent 过滤
    memories = [m for m in memories if m.get('agent_id', 'default') == agent_id]
    
    return jsonify({"memories": memories, "count": len(memories), "agent_id": agent_id})

@app.route('/api/memories', methods=['POST'])
def add_memory():
    """添加记忆"""
    ms = get_memory_system()
    data = request.json
    
    try:
        mem_id = ms.add_memory(
            content=data.get('content'),
            memory_type=data.get('type', 'context'),
            importance=data.get('importance', 0.5),
            tags=data.get('tags', []),
            source=data.get('source', 'api')
        )
        return jsonify({"success": True, "id": mem_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/memories/search', methods=['POST'])
def search_memories():
    """搜索记忆"""
    ms = get_memory_system()
    data = request.json
    
    results = ms.search_memories(
        query=data.get('query', ''),
        limit=data.get('limit', 10),
        days=data.get('days'),
        memory_type=data.get('type')
    )
    return jsonify({"results": results})

@app.route('/api/memories/hybrid-search', methods=['POST'])
def hybrid_search():
    """混合搜索"""
    ms = get_memory_system()
    data = request.json
    
    results = ms.hybrid_search(
        query=data.get('query', ''),
        limit=data.get('limit', 10),
        alpha=data.get('alpha', 0.5)
    )
    return jsonify({"results": results})

@app.route('/api/memories/rerank', methods=['POST'])
def rerank_search():
    """向量搜索 + 重排序"""
    ms = get_memory_system()
    data = request.json
    query = data.get('query', '')
    
    # 先向量搜索
    memories = ms.search_memories(query=query, limit=20)
    
    # 提取文档内容
    documents = [m.get('content', '') for m in memories]
    
    # 重排序
    reranked = ms.rerank_with_dashscope(
        query=query,
        documents=documents,
        top_n=data.get('limit', 5)
    )
    
    # 合并结果
    results = []
    for item in reranked:
        text = item.get('text', '')
        # 找到原始记忆
        for mem in memories:
            if mem.get('content') == text:
                mem['rerank_score'] = item.get('relevance_score', 0)
                results.append(mem)
                break
    
    return jsonify({"results": results})

@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def delete_memory(memory_id):
    """删除记忆"""
    ms = get_memory_system()
    success = ms.delete_memory(memory_id)
    return jsonify({"success": success})

@app.route('/api/entities', methods=['GET'])
def list_entities():
    """获取所有实体"""
    ms = get_memory_system()
    entities = ms.get_all_entities()
    return jsonify({"entities": entities, "count": len(entities)})

@app.route('/api/entities', methods=['POST'])
def add_entity():
    """添加实体"""
    ms = get_memory_system()
    data = request.json
    
    try:
        entity_id = ms.add_entity(
            name=data.get('name'),
            entity_type=data.get('entityType', 'concept'),
            description=data.get('description', '')
        )
        return jsonify({"success": True, "id": entity_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/entities/search', methods=['POST'])
def search_entities():
    """搜索实体"""
    ms = get_memory_system()
    data = request.json
    
    results = ms.search_entities(
        query=data.get('query', ''),
        limit=data.get('limit', 10)
    )
    return jsonify({"results": results})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计"""
    ms = get_memory_system()
    stats = ms.get_memory_stats()
    
    # 获取实体统计
    entities = ms.get_all_entities()
    stats['total_entities'] = len(entities)
    
    # 按类型统计
    type_counts = {}
    memories = ms.get_all_memories(500)
    for mem in memories:
        t = mem.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    stats['by_type'] = type_counts
    
    return jsonify(stats)

# ===== 知识图谱 API =====

@app.route('/api/graph', methods=['GET'])
def get_graph():
    """获取知识图谱数据"""
    import sys
    sys.path.insert(0, BASE_DIR)
    from scripts.knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph()
    data = kg.get_graph_data()
    return jsonify(data)

@app.route('/api/memories/time-query', methods=['POST'])
def time_query():
    """时间范围查询"""
    from scripts.time_range_query import TimeRangeMemoryQuery
    
    data = request.json
    time_expr = data.get('time_expression', '')
    keywords = data.get('keywords', None)
    
    query = TimeRangeMemoryQuery(use_llm=True)
    result = query.query_by_time(time_expr, keywords)
    
    # 合并结果
    memories = []
    for m in result.get('daily_memories', []):
        memories.append({
            'content': m.get('content'),
            'date': m.get('date'),
            'type': m.get('section', 'context'),
            'source': 'daily'
        })
    for m in result.get('neural_memories', []):
        memories.append({
            'content': m.get('content'),
            'date': m.get('date'),
            'type': m.get('type', 'context'),
            'importance': m.get('importance'),
            'source': 'neural'
        })
    
    return jsonify({
        'time_range': result.get('time_range', ''),
        'memories': memories,
        'total_count': len(memories)
    })

@app.route('/api/graph/entity/<entity_name>', methods=['GET'])
def get_entity_relations(entity_name):
    """获取实体关系"""
    import sys
    sys.path.insert(0, BASE_DIR)
    from scripts.knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph()
    related = kg.find_related_entities(entity_name)
    timeline = kg.get_entity_timeline(entity_name)
    
    return jsonify({
        'entity': entity_name,
        'related': related,
        'timeline': timeline[:10]
    })

if __name__ == '__main__':
    print("🚀 记忆系统 Web 服务启动中...")
    print("📍 API: http://localhost:5001/api")
    print("📍 UI: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)