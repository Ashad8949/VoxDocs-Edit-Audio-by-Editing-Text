"""Pytest configuration and fixtures."""

import pytest

from projects.models import Project


@pytest.fixture
def project(db):
    """Create a test project."""
    return Project.objects.create(
        name="Test Project",
        language="en",
        status="ready",
    )


@pytest.fixture
def project_with_video(db):
    """Create a test project with video."""
    project = Project.objects.create(
        name="Test Project with Video",
        language="en",
        status="ready",
        has_video=True,
    )
    return project
