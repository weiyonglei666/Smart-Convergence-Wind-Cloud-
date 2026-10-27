import pymysql

from app.config import get_settings


def mysql_connection(database: str | None = None, connect_timeout: int = 15):
    settings = get_settings()
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database if database is None else database,
        charset="utf8mb4",
        connect_timeout=connect_timeout,
        read_timeout=30,
        write_timeout=30,
        autocommit=True,
    )
