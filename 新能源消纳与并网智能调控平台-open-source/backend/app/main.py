import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.api.routes import router
from app.api.admin_routes import router as admin_router
from app.api.auth_routes import router as auth_router
from app.api.assistant_routes import router as assistant_router
from app.nwp.scheduler import nwp_refresh_loop
from app.nwp.scheduler import next_refresh


@asynccontextmanager
async def lifespan(_: FastAPI):
    refresh_task = asyncio.create_task(nwp_refresh_loop(), name="nwp-refresh-scheduler")
    try:
        yield
    finally:
        refresh_task.cancel()
        await asyncio.gather(refresh_task, return_exceptions=True)

app = FastAPI(title="新能源消纳与并网智能调控平台", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(assistant_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "timezone": settings.timezone_name,
        "nwp_ssh_host": settings.nwp_ssh_host,
        "nwp_remote_dir": settings.nwp_remote_dir,
        "nwp_refresh_hours": settings.nwp_refresh_hours,
        "next_nwp_refresh": next_refresh().isoformat(),
    }
