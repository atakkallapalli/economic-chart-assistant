#!/usr/bin/env python3
"""
Database initialization script for Economic Chart Assistant
Run this once to set up the SQLite database
"""

from chart_database import get_database
import os

def init_database():
    """Initialize the database with tables"""
    print("Initializing Economic Chart Assistant database...")
    
    # Get database instance (this will create tables if they don't exist)
    db = get_database()
    
    # Get statistics to verify setup
    stats = db.get_statistics()
    
    print("Database initialized successfully!")
    print(f"Total charts: {stats['total_charts']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Database file: charts.db")
    
    return True

if __name__ == "__main__":
    init_database()