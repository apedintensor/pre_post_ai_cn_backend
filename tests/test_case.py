# backend/tests/test_case.py
import pytest
from httpx import AsyncClient
from fastapi import status


@pytest.mark.asyncio
async def test_create_diagnosis_term(client: AsyncClient, user_token_headers: dict):
    """Test creating a diagnosis term (needed for case creation)."""
    # Use unique name with timestamp to avoid conflicts
    import time
    unique_name = f"Test Diagnosis Term {time.time()}"
    
    diagnosis_term_data = {
        "name": unique_name
    }
    
    response = await client.post(
        "/diagnosis_terms/",
        headers=user_token_headers,
        json=diagnosis_term_data
    )
    assert response.status_code == 201, f"Failed to create diagnosis term: {response.text}"
    data = response.json()
    assert data["name"] == diagnosis_term_data["name"]
    assert "id" in data
    
    return data


@pytest.mark.asyncio
async def test_create_case(client: AsyncClient, user_token_headers: dict):
    """Test creating a new case."""
    # First create a diagnosis term
    diagnosis_term = await test_create_diagnosis_term(client, user_token_headers)
    
    # Now create a case
    case_data = {
        "ground_truth_diagnosis_id": diagnosis_term["id"],
        "typical_diagnosis": True
    }
    
    response = await client.post(
        "/cases/", 
        headers=user_token_headers,
        json=case_data
    )
    assert response.status_code == 201, f"Failed to create case: {response.text}"
    data = response.json()
    assert "id" in data, "Case id not found in response"
    assert data["ground_truth_diagnosis_id"] == diagnosis_term["id"]
    assert data["typical_diagnosis"] == case_data["typical_diagnosis"]
    
    # Create metadata for the case
    metadata_data = {
        "age": 45,
        "gender": "male",
        "fever_history": False,
        "psoriasis_history": True,
        "other_notes": "Test case metadata"
    }
    
    metadata_response = await client.post(
        f"/case_metadata/case/{data['id']}", 
        headers=user_token_headers,
        json=metadata_data
    )
    assert metadata_response.status_code == 201, f"Failed to create case metadata: {metadata_response.text}"
    
    return data


@pytest.mark.asyncio
async def test_get_case(client: AsyncClient, user_token_headers: dict):
    """Test retrieving a case by ID."""
    # First create a case
    case = await test_create_case(client, user_token_headers)
    
    # Now retrieve it
    response = await client.get(
        f"/cases/{case['id']}", 
        headers=user_token_headers
    )
    assert response.status_code == 200, f"Failed to get case: {response.text}"
    data = response.json()
    assert data["id"] == case["id"]
    assert data["ground_truth_diagnosis_id"] == case["ground_truth_diagnosis_id"]
    assert data["typical_diagnosis"] == case["typical_diagnosis"]


@pytest.mark.asyncio
async def test_get_cases(client: AsyncClient, user_token_headers: dict):
    """Test retrieving list of cases."""
    # Create a case first
    await test_create_case(client, user_token_headers)
    
    response = await client.get(
        "/cases/", 
        headers=user_token_headers
    )
    assert response.status_code == 200, f"Failed to get cases: {response.text}"
    data = response.json()
    assert isinstance(data, list), "Expected a list of cases"
    assert len(data) > 0, "Expected at least one case"
    assert "id" in data[0], "Case id not found in response"


@pytest.mark.asyncio
async def test_get_case_unauthorized(client: AsyncClient):
    """Test that getting case without auth token fails."""
    response = await client.get("/cases/1")
    print(f"Unauthorized test response: {response.status_code} - {response.text}")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED, "Should fail without token"


@pytest.mark.asyncio
async def test_create_case_with_invalid_data(client: AsyncClient, user_token_headers: dict):
    """Test validation for case creation with missing required fields."""
    # Missing ground_truth_diagnosis_id
    invalid_case_data = {
        "typical_diagnosis": True
    }
    
    response = await client.post(
        "/cases/", 
        headers=user_token_headers,
        json=invalid_case_data
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, "Should fail validation"


@pytest.mark.asyncio
async def test_get_nonexistent_case(client: AsyncClient, user_token_headers: dict):
    """Test getting a case that doesn't exist."""
    response = await client.get(
        "/cases/99999", 
        headers=user_token_headers
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, "Should return 404 for nonexistent case"
    assert "not found" in response.text.lower(), "Error message should mention not found"


@pytest.mark.asyncio
async def test_get_case_images(client: AsyncClient, user_token_headers: dict):
    """Test retrieving images for a case."""
    # First create a case
    case = await test_create_case(client, user_token_headers)
    
    response = await client.get(
        f"/images/case/{case['id']}", 
        headers=user_token_headers
    )
    assert response.status_code == 200, f"Failed to get case images: {response.text}"
    data = response.json()
    assert isinstance(data, list), "Expected a list of images"
    # Note: The list might be empty if no images were added during case creation


@pytest.mark.asyncio
async def test_get_management_strategies(client: AsyncClient, user_token_headers: dict):
    """Test retrieving management strategies."""
    response = await client.get(
        "/management_strategies/",
        headers=user_token_headers
    )
    assert response.status_code == 200, f"Failed to get management strategies: {response.text}"
    data = response.json()
    assert isinstance(data, list), "Expected a list of management strategies"
    assert len(data) > 0, "Expected at least one management strategy"
    assert "id" in data[0], "Strategy id not found in response"
    assert "name" in data[0], "Strategy name not found in response"


@pytest.mark.asyncio
async def test_get_case_metadata(client: AsyncClient, user_token_headers: dict):
    """Test retrieving metadata for a case."""
    # First create a case
    case = await test_create_case(client, user_token_headers)
    
    response = await client.get(
        f"/case_metadata/case/{case['id']}",
        headers=user_token_headers
    )
    assert response.status_code == 200, f"Failed to get case metadata: {response.text}"
    data = response.json()
    assert "id" in data, "Metadata id not found in response"
    assert "case_id" in data, "Case id not found in response"
    assert data["case_id"] == case["id"], "Case id mismatch"