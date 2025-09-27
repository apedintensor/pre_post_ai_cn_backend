# backend/tests/test_auth.py
import pytest
from httpx import AsyncClient
from fastapi import status


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    """Test user registration."""
    new_user = {
        "email": "newuser@example.com",
        "password": "securepass123",
        "is_active": True,
        "role_id": 1,
    }
    
    response = await client.post("/auth/register/register", json=new_user)
    assert response.status_code == 201, f"Registration failed: {response.text}"
    data = response.json()
    assert data["email"] == new_user["email"]
    assert "id" in data, "User id not found in response"
    assert "hashed_password" not in data, "Password should not be returned"


@pytest.mark.asyncio
async def test_register_existing_user_fails(client: AsyncClient, registered_user: dict):
    """Test that registering an existing user fails."""
    response = await client.post("/auth/register/register", json=registered_user)
    assert response.status_code == status.HTTP_400_BAD_REQUEST, "Should fail with duplicate email"
    # Updated assertion to match the actual error format
    error_data = response.json()
    assert "detail" in error_data, "Error response should contain detail field"
    assert "register_user_already_exists" in error_data["detail"].lower(), "Error message should mention user already exists"


@pytest.mark.asyncio
async def test_login(client: AsyncClient, registered_user: dict):
    """Test user login with valid credentials."""
    login_data = {
        "username": registered_user["email"],
        "password": registered_user["password"],
    }
    
    response = await client.post("/auth/jwt/login", data=login_data)
    assert response.status_code == 200, f"Login failed: {response.text}"
    data = response.json()
    assert "access_token" in data, "No access token returned"
    assert "token_type" in data, "No token type returned"
    assert data["token_type"] == "bearer", "Token type is not bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    """Test login with invalid credentials."""
    login_data = {
        "username": "fake@example.com",
        "password": "wrongpassword",
    }
    
    response = await client.post("/auth/jwt/login", data=login_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST, "Should fail with invalid credentials"


@pytest.mark.asyncio
async def test_get_user_me(client: AsyncClient, user_token_headers: dict):
    """Test getting user profile with valid JWT token."""
    response = await client.get("/auth/me", headers=user_token_headers)
    assert response.status_code == 200, f"Failed to get user profile: {response.text}"
    data = response.json()
    assert "email" in data, "Email not found in user profile"
    assert "id" in data, "User id not found in user profile"


@pytest.mark.asyncio
async def test_get_user_me_without_token(client: AsyncClient):
    """Test that getting user profile without token fails."""
    response = await client.get("/auth/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED, "Should fail without token"
    assert "unauthorized" in response.text.lower(), "Error message should mention unauthorized"