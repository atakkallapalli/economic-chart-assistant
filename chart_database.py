"""
Chart Database Module
Handles persistence of generated charts using SQLite
"""
import sqlite3
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

class ChartDatabase:
    """Database handler for chart persistence"""
    
    def __init__(self, db_path: str = "charts.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Charts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS charts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chart_name TEXT NOT NULL,
                chart_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_prompt TEXT,
                chart_config TEXT,
                figure_json TEXT,
                metadata TEXT,
                tags TEXT
            )
        ''')
        
        # Chart data table (for storing associated data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chart_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chart_id INTEGER,
                data_type TEXT,
                data_content BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chart_id) REFERENCES charts (id) ON DELETE CASCADE
            )
        ''')
        
        # Chart sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chart_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        ''')
        
        # Chart-session mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_charts (
                session_id INTEGER,
                chart_id INTEGER,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chart_sessions (id) ON DELETE CASCADE,
                FOREIGN KEY (chart_id) REFERENCES charts (id) ON DELETE CASCADE,
                PRIMARY KEY (session_id, chart_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_chart(self, chart_name: str, chart_type: str, 
                   figure_json: str, chart_config: Optional[Dict] = None,
                   user_prompt: Optional[str] = None, 
                   metadata: Optional[Dict] = None,
                   tags: Optional[List[str]] = None) -> int:
        """
        Save a chart to the database
        
        Args:
            chart_name: Name of the chart
            chart_type: Type (custom, precreated, etc.)
            figure_json: Plotly figure as JSON string
            chart_config: Chart configuration dict
            user_prompt: Original user prompt
            metadata: Additional metadata
            tags: List of tags for categorization
        
        Returns:
            Chart ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO charts (chart_name, chart_type, user_prompt, chart_config, 
                              figure_json, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            chart_name,
            chart_type,
            user_prompt,
            json.dumps(chart_config) if chart_config else None,
            figure_json,
            json.dumps(metadata) if metadata else None,
            json.dumps(tags) if tags else None
        ))
        
        chart_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return chart_id
    
    def get_chart(self, chart_id: int) -> Optional[Dict]:
        """Retrieve a chart by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, chart_name, chart_type, created_at, updated_at,
                   user_prompt, chart_config, figure_json, metadata, tags
            FROM charts
            WHERE id = ?
        ''', (chart_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'chart_name': row[1],
                'chart_type': row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'user_prompt': row[5],
                'chart_config': json.loads(row[6]) if row[6] else None,
                'figure_json': row[7],
                'metadata': json.loads(row[8]) if row[8] else None,
                'tags': json.loads(row[9]) if row[9] else None
            }
        return None
    
    def list_charts(self, chart_type: Optional[str] = None, 
                   limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all charts with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if chart_type:
            cursor.execute('''
                SELECT id, chart_name, chart_type, created_at, user_prompt, tags
                FROM charts
                WHERE chart_type = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (chart_type, limit, offset))
        else:
            cursor.execute('''
                SELECT id, chart_name, chart_type, created_at, user_prompt, tags
                FROM charts
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        charts = []
        for row in rows:
            charts.append({
                'id': row[0],
                'chart_name': row[1],
                'chart_type': row[2],
                'created_at': row[3],
                'user_prompt': row[4],
                'tags': json.loads(row[5]) if row[5] else None
            })
        
        return charts
    
    def update_chart(self, chart_id: int, **kwargs):
        """Update chart fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['chart_name', 'chart_type', 'user_prompt']:
                update_fields.append(f"{key} = ?")
                values.append(value)
            elif key in ['chart_config', 'metadata', 'tags']:
                update_fields.append(f"{key} = ?")
                values.append(json.dumps(value) if value else None)
            elif key == 'figure_json':
                update_fields.append(f"{key} = ?")
                values.append(value)
        
        if update_fields:
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(chart_id)
            
            query = f"UPDATE charts SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def delete_chart(self, chart_id: int):
        """Delete a chart"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM charts WHERE id = ?', (chart_id,))
        conn.commit()
        conn.close()
    
    def search_charts(self, search_term: str) -> List[Dict]:
        """Search charts by name or prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, chart_name, chart_type, created_at, user_prompt, tags
            FROM charts
            WHERE chart_name LIKE ? OR user_prompt LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{search_term}%', f'%{search_term}%'))
        
        rows = cursor.fetchall()
        conn.close()
        
        charts = []
        for row in rows:
            charts.append({
                'id': row[0],
                'chart_name': row[1],
                'chart_type': row[2],
                'created_at': row[3],
                'user_prompt': row[4],
                'tags': json.loads(row[5]) if row[5] else None
            })
        
        return charts
    
    def save_chart_data(self, chart_id: int, data_type: str, data: Any):
        """Save associated data for a chart (e.g., DataFrame)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize data
        data_blob = pickle.dumps(data)
        
        cursor.execute('''
            INSERT INTO chart_data (chart_id, data_type, data_content)
            VALUES (?, ?, ?)
        ''', (chart_id, data_type, data_blob))
        
        conn.commit()
        conn.close()
    
    def get_chart_data(self, chart_id: int, data_type: str) -> Optional[Any]:
        """Retrieve associated data for a chart"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_content
            FROM chart_data
            WHERE chart_id = ? AND data_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (chart_id, data_type))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return pickle.loads(row[0])
        return None
    
    def create_session(self, session_name: str, description: Optional[str] = None) -> int:
        """Create a new chart session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chart_sessions (session_name, description)
            VALUES (?, ?)
        ''', (session_name, description))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_chart_to_session(self, session_id: int, chart_id: int):
        """Add a chart to a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO session_charts (session_id, chart_id)
            VALUES (?, ?)
        ''', (session_id, chart_id))
        
        conn.commit()
        conn.close()
    
    def get_session_charts(self, session_id: int) -> List[Dict]:
        """Get all charts in a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.chart_name, c.chart_type, c.created_at, c.user_prompt
            FROM charts c
            JOIN session_charts sc ON c.id = sc.chart_id
            WHERE sc.session_id = ?
            ORDER BY sc.added_at DESC
        ''', (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        charts = []
        for row in rows:
            charts.append({
                'id': row[0],
                'chart_name': row[1],
                'chart_type': row[2],
                'created_at': row[3],
                'user_prompt': row[4]
            })
        
        return charts
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total charts
        cursor.execute('SELECT COUNT(*) FROM charts')
        total_charts = cursor.fetchone()[0]
        
        # Charts by type
        cursor.execute('''
            SELECT chart_type, COUNT(*) 
            FROM charts 
            GROUP BY chart_type
        ''')
        charts_by_type = dict(cursor.fetchall())
        
        # Total sessions
        cursor.execute('SELECT COUNT(*) FROM chart_sessions')
        total_sessions = cursor.fetchone()[0]
        
        # Recent charts
        cursor.execute('''
            SELECT COUNT(*) 
            FROM charts 
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        recent_charts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_charts': total_charts,
            'charts_by_type': charts_by_type,
            'total_sessions': total_sessions,
            'recent_charts': recent_charts
        }
    
    def cleanup_old_charts(self, days: int = 30):
        """Delete charts older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM charts 
            WHERE created_at < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    def export_charts(self, output_dir: str = "exported_charts"):
        """Export all charts to files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = self.list_charts(limit=1000)
        
        for chart in charts:
            full_chart = self.get_chart(chart['id'])
            if full_chart and full_chart['figure_json']:
                filename = f"{output_dir}/chart_{chart['id']}_{chart['chart_name']}.json"
                with open(filename, 'w') as f:
                    f.write(full_chart['figure_json'])
        
        return len(charts)


# Singleton instance
_db_instance = None

def get_database() -> ChartDatabase:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ChartDatabase()
    return _db_instance
