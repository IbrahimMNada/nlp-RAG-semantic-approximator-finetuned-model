"""
Consolidated models file for Alembic migrations.
This file imports all models to ensure they're registered with SQLAlchemy metadata.
"""
import sys
from pathlib import Path
from src.core.base import Base

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


from src.modules.data.entities import *    # register the data module models



target_metadata = Base.metadata