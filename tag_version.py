#!/usr/bin/env python3
"""
Create git tags for semantic versioning with setuptools-scm
"""
import re
import sys
import subprocess

def get_latest_tag():
    """Get the latest git tag that looks like a version"""
    try:
        # Get all version tags and sort them
        result = subprocess.run(
            ['git', 'tag', '-l', 'v*.*.*'],
            capture_output=True,
            text=True,
            check=True
        )
        tags = result.stdout.strip().split('\n')
        if not tags or tags == ['']:
            return "v1.0.7"  # Start from current version
        
        # Sort tags by version (simple string sort works for semantic versions)
        version_tags = [tag for tag in tags if tag.startswith('v') and tag.count('.') == 2]
        if not version_tags:
            return "v1.0.7"
            
        # Sort and get the latest
        version_tags.sort(key=lambda x: [int(i) for i in x[1:].split('.')])
        return version_tags[-1]
        
    except subprocess.CalledProcessError:
        # No tags found, start with current version
        return "v1.0.7"

def get_latest_commit_message():
    """Get the latest commit message"""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--pretty=%B'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def parse_version_type_from_commit(commit_message):
    """
    Parse commit message to determine version bump type.
    
    Examples:
    - "patch: something" -> patch
    - "minor: new feature" -> minor
    - "major: breaking change" -> major
    - "random commit" -> patch (default)
    """
    if not commit_message:
        return "patch"
    
    normalized = commit_message.strip().lower()
    
    patterns = {
        "major": r'^\s*major\s*[:\-]',
        "minor": r'^\s*minor\s*[:\-]',
        "patch": r'^\s*patch\s*[:\-]',
    }
    
    if re.match(patterns["major"], normalized):
        return "major"
    elif re.match(patterns["minor"], normalized):
        return "minor"
    else:
        return "patch"

def bump_version_tag(current_tag, bump_type):
    """Increment version tag based on bump type"""
    # Remove 'v' prefix if present
    version_str = current_tag.lstrip('v')
    
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major, minor, patch = map(int, parts)
    
    if bump_type == "major":
        new_version = f"v{major + 1}.0.0"
    elif bump_type == "minor":
        new_version = f"v{major}.{minor + 1}.0"
    else:  # patch
        new_version = f"v{major}.{minor}.{patch + 1}"
    
    return new_version

def create_git_tag(tag_name, message):
    """Create and push a git tag"""
    try:
        # Create annotated tag
        subprocess.run(
            ['git', 'tag', '-a', tag_name, '-m', message],
            check=True
        )
        
        # Push tag to remote
        subprocess.run(
            ['git', 'push', 'origin', tag_name],
            check=True
        )
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating/pushing tag: {e}", file=sys.stderr)
        return False

def main():
    try:
        current_tag = get_latest_tag()
        commit_message = get_latest_commit_message()
        bump_type = parse_version_type_from_commit(commit_message)
        new_tag = bump_version_tag(current_tag, bump_type)
        
        tag_message = f"Release {new_tag}"
        if create_git_tag(new_tag, tag_message):
            print(f"Commit: {commit_message[:50]}{'...' if len(commit_message) > 50 else ''}")
            print(f"Bump type: {bump_type}")
            print(f"Version tag: {current_tag} -> {new_tag}")
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())