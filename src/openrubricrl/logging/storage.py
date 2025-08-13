"""Storage backends for logging scoring results."""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.reward_processor import ProcessedReward


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store_result(self, result: ProcessedReward, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a scoring result and return a unique identifier."""
        pass
    
    @abstractmethod
    def get_result(self, result_id: str) -> Optional[ProcessedReward]:
        """Retrieve a result by ID."""
        pass
    
    @abstractmethod
    def query_results(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query results with optional filters."""
        pass
    
    @abstractmethod
    def get_statistics(self, rubric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        pass


class SQLiteStorage(StorageBackend):
    """SQLite storage backend for results."""
    
    def __init__(self, db_path: str = "openrubricrl.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scoring_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    rubric_name TEXT NOT NULL,
                    rubric_version TEXT NOT NULL,
                    task_input TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    reward REAL NOT NULL,
                    raw_llm_score REAL NOT NULL,
                    normalized_score REAL NOT NULL,
                    explanation TEXT,
                    hybrid_components TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rubric_name ON scoring_results(rubric_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON scoring_results(timestamp)
            """)
            
            conn.commit()
    
    def store_result(self, result: ProcessedReward, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a result in SQLite."""
        timestamp = datetime.now().isoformat()
        
        # Extract metadata
        meta = result.metadata or {}
        rubric_name = meta.get("rubric_name", "unknown")
        rubric_version = meta.get("rubric_version", "unknown")
        
        # Get task input/output from metadata if available
        task_input = metadata.get("task_input", "") if metadata else ""
        model_output = metadata.get("model_output", "") if metadata else ""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO scoring_results (
                    timestamp, rubric_name, rubric_version, task_input, model_output,
                    reward, raw_llm_score, normalized_score, explanation,
                    hybrid_components, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                rubric_name,
                rubric_version,
                task_input,
                model_output,
                result.reward,
                result.raw_llm_score,
                result.normalized_score,
                result.explanation,
                json.dumps(result.hybrid_components),
                json.dumps(metadata or {})
            ))
            
            return str(cursor.lastrowid)
    
    def get_result(self, result_id: str) -> Optional[ProcessedReward]:
        """Retrieve a result by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM scoring_results WHERE id = ?
            """, (result_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return ProcessedReward(
                reward=row["reward"],
                raw_llm_score=row["raw_llm_score"],
                normalized_score=row["normalized_score"],
                hybrid_components=json.loads(row["hybrid_components"]),
                explanation=row["explanation"],
                metadata=json.loads(row["metadata"])
            )
    
    def query_results(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query results with filters."""
        query = "SELECT * FROM scoring_results"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if key in ["rubric_name", "rubric_version"]:
                    conditions.append(f"{key} = ?")
                    params.append(value)
                elif key == "min_reward":
                    conditions.append("reward >= ?")
                    params.append(value)
                elif key == "max_reward":
                    conditions.append("reward <= ?")
                    params.append(value)
                elif key == "after":
                    conditions.append("timestamp >= ?")
                    params.append(value)
                elif key == "before":
                    conditions.append("timestamp <= ?")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self, rubric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        query = """
            SELECT 
                COUNT(*) as total_results,
                AVG(reward) as avg_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                AVG(raw_llm_score) as avg_llm_score,
                MIN(timestamp) as earliest_result,
                MAX(timestamp) as latest_result
            FROM scoring_results
        """
        params = []
        
        if rubric_name:
            query += " WHERE rubric_name = ?"
            params.append(rubric_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                return {
                    "total_results": result[0],
                    "avg_reward": result[1] or 0.0,
                    "min_reward": result[2] or 0.0,
                    "max_reward": result[3] or 0.0,
                    "avg_llm_score": result[4] or 0.0,
                    "earliest_result": result[5],
                    "latest_result": result[6],
                    "rubric_name": rubric_name
                }
            
            return {}


class JSONFileStorage(StorageBackend):
    """JSON file storage backend."""
    
    def __init__(self, file_path: str = "openrubricrl_results.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.touch(exist_ok=True)
    
    def store_result(self, result: ProcessedReward, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store result in JSONL format."""
        timestamp = datetime.now().isoformat()
        
        record = {
            "id": timestamp,  # Use timestamp as ID for simplicity
            "timestamp": timestamp,
            "reward": result.reward,
            "raw_llm_score": result.raw_llm_score,
            "normalized_score": result.normalized_score,
            "explanation": result.explanation,
            "hybrid_components": result.hybrid_components,
            "metadata": {**(result.metadata or {}), **(metadata or {})}
        }
        
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return timestamp
    
    def get_result(self, result_id: str) -> Optional[ProcessedReward]:
        """Retrieve result by ID."""
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record["id"] == result_id:
                        return ProcessedReward(
                            reward=record["reward"],
                            raw_llm_score=record["raw_llm_score"],
                            normalized_score=record["normalized_score"],
                            hybrid_components=record["hybrid_components"],
                            explanation=record["explanation"],
                            metadata=record["metadata"]
                        )
        return None
    
    def query_results(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query results from JSONL file."""
        results = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    
                    # Apply filters
                    if filters:
                        skip = False
                        for key, value in filters.items():
                            if key == "rubric_name" and record["metadata"].get("rubric_name") != value:
                                skip = True
                                break
                            elif key == "min_reward" and record["reward"] < value:
                                skip = True
                                break
                            elif key == "max_reward" and record["reward"] > value:
                                skip = True
                                break
                        
                        if skip:
                            continue
                    
                    results.append(record)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_statistics(self, rubric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        results = self.query_results(
            filters={"rubric_name": rubric_name} if rubric_name else None
        )
        
        if not results:
            return {}
        
        rewards = [r["reward"] for r in results]
        llm_scores = [r["raw_llm_score"] for r in results]
        timestamps = [r["timestamp"] for r in results]
        
        return {
            "total_results": len(results),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_llm_score": sum(llm_scores) / len(llm_scores),
            "earliest_result": min(timestamps),
            "latest_result": max(timestamps),
            "rubric_name": rubric_name
        }


class PostgreSQLStorage(StorageBackend):
    """PostgreSQL storage backend (requires psycopg2)."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "openrubricrl",
        user: str = "postgres",
        password: str = "",
        table_name: str = "scoring_results"
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self.table_name = table_name
        self._init_database()
    
    def _get_connection(self):
        """Get database connection."""
        try:
            import psycopg2
            return psycopg2.connect(**self.connection_params)
        except ImportError:
            raise ImportError("Install psycopg2: pip install psycopg2-binary")
    
    def _init_database(self):
        """Initialize PostgreSQL database schema."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        rubric_name VARCHAR(255) NOT NULL,
                        rubric_version VARCHAR(50) NOT NULL,
                        task_input TEXT NOT NULL,
                        model_output TEXT NOT NULL,
                        reward FLOAT NOT NULL,
                        raw_llm_score FLOAT NOT NULL,
                        normalized_score FLOAT NOT NULL,
                        explanation TEXT,
                        hybrid_components JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_rubric_name 
                    ON {self.table_name}(rubric_name)
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                    ON {self.table_name}(timestamp)
                """)
                
                conn.commit()
    
    def store_result(self, result: ProcessedReward, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store result in PostgreSQL."""
        timestamp = datetime.now()
        
        # Extract metadata
        meta = result.metadata or {}
        rubric_name = meta.get("rubric_name", "unknown")
        rubric_version = meta.get("rubric_version", "unknown")
        
        task_input = metadata.get("task_input", "") if metadata else ""
        model_output = metadata.get("model_output", "") if metadata else ""
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {self.table_name} (
                        timestamp, rubric_name, rubric_version, task_input, model_output,
                        reward, raw_llm_score, normalized_score, explanation,
                        hybrid_components, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    rubric_name,
                    rubric_version,
                    task_input,
                    model_output,
                    result.reward,
                    result.raw_llm_score,
                    result.normalized_score,
                    result.explanation,
                    json.dumps(result.hybrid_components),
                    json.dumps(metadata or {})
                ))
                
                result_id = cursor.fetchone()[0]
                conn.commit()
                return str(result_id)
    
    def get_result(self, result_id: str) -> Optional[ProcessedReward]:
        """Retrieve result by ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT reward, raw_llm_score, normalized_score, explanation,
                           hybrid_components, metadata
                    FROM {self.table_name} WHERE id = %s
                """, (result_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return ProcessedReward(
                    reward=row[0],
                    raw_llm_score=row[1],
                    normalized_score=row[2],
                    hybrid_components=row[4],
                    explanation=row[3],
                    metadata=row[5]
                )
    
    def query_results(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query results with filters."""
        query = f"SELECT * FROM {self.table_name}"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if key in ["rubric_name", "rubric_version"]:
                    conditions.append(f"{key} = %s")
                    params.append(value)
                elif key == "min_reward":
                    conditions.append("reward >= %s")
                    params.append(value)
                elif key == "max_reward":
                    conditions.append("reward <= %s")
                    params.append(value)
                elif key == "after":
                    conditions.append("timestamp >= %s")
                    params.append(value)
                elif key == "before":
                    conditions.append("timestamp <= %s")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
    
    def get_statistics(self, rubric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        query = f"""
            SELECT 
                COUNT(*) as total_results,
                AVG(reward) as avg_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                AVG(raw_llm_score) as avg_llm_score,
                MIN(timestamp) as earliest_result,
                MAX(timestamp) as latest_result
            FROM {self.table_name}
        """
        params = []
        
        if rubric_name:
            query += " WHERE rubric_name = %s"
            params.append(rubric_name)
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    return {
                        "total_results": result[0],
                        "avg_reward": float(result[1]) if result[1] else 0.0,
                        "min_reward": float(result[2]) if result[2] else 0.0,
                        "max_reward": float(result[3]) if result[3] else 0.0,
                        "avg_llm_score": float(result[4]) if result[4] else 0.0,
                        "earliest_result": result[5].isoformat() if result[5] else None,
                        "latest_result": result[6].isoformat() if result[6] else None,
                        "rubric_name": rubric_name
                    }
                
                return {}


# Factory function for creating storage backends
def create_storage_backend(
    backend_type: str = "sqlite",
    **kwargs
) -> StorageBackend:
    """Create a storage backend."""
    if backend_type == "sqlite":
        return SQLiteStorage(**kwargs)
    elif backend_type == "jsonl":
        return JSONFileStorage(**kwargs)
    elif backend_type == "postgresql":
        return PostgreSQLStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")


# Global storage instance
_default_storage = None


def get_default_storage() -> StorageBackend:
    """Get the default storage backend."""
    global _default_storage
    if _default_storage is None:
        _default_storage = SQLiteStorage()
    return _default_storage


def set_default_storage(storage: StorageBackend) -> None:
    """Set the default storage backend."""
    global _default_storage
    _default_storage = storage