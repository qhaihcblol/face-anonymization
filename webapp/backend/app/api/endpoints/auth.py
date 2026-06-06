from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest
from app.services.auth_service import AuthService
from app.utils.exceptions import AppException


router = APIRouter()


@router.post(
    "/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    return await AuthService.register(db, payload)


@router.post("/login", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    return await AuthService.login(db, payload)


@router.post("/token", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """OAuth2 password-flow token endpoint backing the Swagger "Authorize" dialog.

    The OAuth2 form's ``username`` field carries the account email; ``client_id`` /
    ``client_secret`` are unused. The frontend uses the JSON ``/login`` endpoint.
    """
    try:
        payload = LoginRequest(
            email=form_data.username, password=form_data.password
        )
    except ValidationError:
        raise AppException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    return await AuthService.login(db, payload)
