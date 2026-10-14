import secrets
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import get_settings

router = APIRouter(prefix="/api/auth")
TOKENS: set[str] = set()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post('/login')
def login(payload: LoginRequest):
    settings = get_settings()
    if payload.username != settings.admin_username or payload.password != settings.admin_password:
        raise HTTPException(status_code=401, detail='用户名或密码错误')
    token = secrets.token_urlsafe(32)
    TOKENS.add(token)
    return {'token': token, 'username': settings.admin_username}
