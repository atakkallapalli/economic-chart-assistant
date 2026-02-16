"""
Database Migration Script
Adds category field, timestamps, unique constraints, and cleans up data
"""
import sqlite3
import json
from datetime import datetime

def migrate_database():
    conn = sqlite3.connect("charts.db")
    cursor = conn.cursor()
    
    print("Starting database migration...")
    
    # 1. Add new columns
    try:
        cursor.execute('ALTER TABLE charts ADD COLUMN category TEXT')
        print("Added category column")
    except:
        print("Category column already exists")
    
    try:
        cursor.execute('ALTER TABLE charts ADD COLUMN created_timestamp TEXT')
        cursor.execute('ALTER TABLE charts ADD COLUMN updated_timestamp TEXT')
        print("Added timestamp columns")
    except:
        print("Timestamp columns already exist")
    
    # 2. Update existing records
    current_time = datetime.now().isoformat()
    
    # Get all charts
    cursor.execute('SELECT id, chart_config, chart_type FROM charts')
    charts = cursor.fetchall()
    
    updated_count = 0
    for chart_id, config_str, chart_type in charts:
        # Extract category from config
        category = None
        if config_str:
            try:
                config = json.loads(config_str)
                category = config.get('report_category', 'Custom Analysis')
            except:
                category = 'Custom Analysis'
        else:
            category = 'Custom Analysis'
        
        # Update chart type to Line
        new_chart_type = 'Line' if chart_type != 'Line' else chart_type
        
        # Update record
        cursor.execute('''
            UPDATE charts 
            SET category = ?, 
                chart_type = ?, 
                created_timestamp = ?, 
                updated_timestamp = ?
            WHERE id = ?
        ''', (category, new_chart_type, current_time, current_time, chart_id))
        
        updated_count += 1
        print(f"Updated chart {chart_id}: category={category}, type={new_chart_type}")
    
    print(f"Updated {updated_count} charts")
    
    # 3. Remove duplicates (keep latest by id)
    cursor.execute('''
        DELETE FROM charts 
        WHERE id NOT IN (
            SELECT MAX(id) 
            FROM charts 
            GROUP BY chart_name, category
        )
    ''')
    
    deleted_count = cursor.rowcount
    print(f"Removed {deleted_count} duplicate charts")
    
    # 4. Create unique constraint (recreate table)
    cursor.execute('''
        CREATE TABLE charts_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chart_name TEXT NOT NULL,
            chart_type TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_timestamp TEXT,
            updated_timestamp TEXT,
            user_prompt TEXT,
            chart_config TEXT,
            figure_json TEXT,
            metadata TEXT,
            tags TEXT,
            UNIQUE(chart_name, category)
        )
    ''')
    
    # Copy data to new table
    cursor.execute('''
        INSERT INTO charts_new 
        SELECT id, chart_name, chart_type, category, created_at, updated_at, 
               created_timestamp, updated_timestamp, user_prompt, chart_config, 
               figure_json, metadata, tags
        FROM charts
    ''')
    
    # Replace old table
    cursor.execute('DROP TABLE charts')
    cursor.execute('ALTER TABLE charts_new RENAME TO charts')
    
    print("Added unique constraint on (chart_name, category)")
    
    conn.commit()
    conn.close()
    
    print("Migration completed successfully!")

if __name__ == "__main__":
    migrate_database()