# backend/tests/test_role.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def create_role(client: AsyncClient, user_token_headers: dict):
    """Helper function to create a test role."""
    import time
    unique_name = f"Test Role {time.time()}"
    role_data = {
        "name": unique_name
    }
    
    response = await client.post(
        "/roles/",
        headers=user_token_headers,
        json=role_data
    )
    assert response.status_code == 201, f"Failed to create role: {response.text}"
    return response.json()

@pytest.mark.asyncio
async def test_read_roles(client: AsyncClient, user_token_headers: dict):
    """Test retrieving roles."""
    response = await client.get(
        "/roles/",
        headers=user_token_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0  # Should have at least one role
    assert all("id" in role and "name" in role for role in data)

@pytest.mark.asyncio
async def test_create_role(client: AsyncClient, user_token_headers: dict):
    """Test creating a new role."""
    role = await create_role(client, user_token_headers)
    assert "id" in role
    assert "name" in role

@pytest.mark.asyncio
async def test_get_role(client: AsyncClient, user_token_headers: dict):
    """Test retrieving a specific role by ID."""
    # First create a role
    role = await create_role(client, user_token_headers)
    
    # Now retrieve it
    response = await client.get(
        f"/roles/{role['id']}",
        headers=user_token_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == role["id"]
    assert data["name"] == role["name"]

@pytest.mark.asyncio
async def test_get_nonexistent_role(client: AsyncClient, user_token_headers: dict):
    """Test retrieving a role that doesn't exist."""
    response = await client.get(
        "/roles/999999",
        headers=user_token_headers
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Role not found"