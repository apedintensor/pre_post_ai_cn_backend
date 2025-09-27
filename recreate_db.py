# recreate_db.py
import asyncio
from app.db.base import Base
from app.db.session import sync_engine, async_engine
from app.db.init_db import init_db

async def main():
    print("Dropping all tables...")
    Base.metadata.drop_all(sync_engine)
    print("Creating all tables...")
    Base.metadata.create_all(sync_engine)
    print("Initializing database with required data...")
    await init_db()
    print("Database has been recreated with the current schema.")

if __name__ == "__main__":
    asyncio.run(main())