#!/usr/bin/env python3
"""
AI-powered conflict resolution script for backport PRs.

This script identifies git conflict markers in files and uses Claude AI
to intelligently resolve each conflict region.
"""

import anthropic
import subprocess
import os
import sys
import re


def extract_conflicts(content):
    """Extract all conflict regions from content"""
    pattern = r'(<<<<<<< HEAD\n)(.*?)(=======\n)(.*?)(>>>>>>> .*?\n)'
    conflicts = []
    for match in re.finditer(pattern, content, re.DOTALL):
        conflicts.append({
            'full_match': match.group(0),
            'head_version': match.group(2),
            'incoming_version': match.group(4),
            'start': match.start(),
            'end': match.end()
        })
    return conflicts


def resolve_conflict_with_ai(client, conflict, filepath, branch, index, total):
    """Use AI to resolve a single conflict region"""
    print(f"\n  Conflict {index}/{total}:")
    print(f"    HEAD version: {len(conflict['head_version'])} chars")
    print(f"    Incoming version: {len(conflict['incoming_version'])} chars")
    
    prompt = f"""You are resolving ONE git merge conflict in a backport PR.

**Context:**
- Backporting to `{branch}`
- This is conflict region {index} of {total} in file `{filepath}`

**HEAD version (what's in the release branch):**
```
{conflict['head_version']}
```

**Incoming version (what the backport is trying to apply):**
```
{conflict['incoming_version']}
```

**Your task:**
Provide the RESOLVED content for this conflict region only.

**Rules:**
1. Return ONLY the resolved content (no markers, no explanations)
2. If incoming removes code, prefer the removal unless release branch has critical differences
3. If incoming adds code, include it unless it conflicts with release-specific changes
4. Keep important comments from the release branch
5. No markdown blocks, just the raw resolved content

Return ONLY the resolved content for this section:
"""

    print(f"    📤 Sending to AI... (prompt: {len(prompt)} chars)")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    resolution = response.content[0].text.strip()
    
    # Clean up markdown if present
    if resolution.startswith("```"):
        lines = resolution.split('\n')
        resolution = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
        resolution = resolution.strip()
    
    print(f"    📥 Received: {len(resolution)} chars, {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")
    print(f"    Resolution preview: {resolution[:100]}{'...' if len(resolution) > 100 else ''}")
    
    return resolution


def resolve_file_conflicts(client, filepath, base_branch):
    """Resolve all conflicts in a file"""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print('='*60)
    
    # Read the conflicted file content
    with open(filepath, 'r') as f:
        conflicted_content = f.read()
    
    # Extract conflict regions
    conflicts = extract_conflicts(conflicted_content)
    print(f"Found {len(conflicts)} conflict region(s) in file")
    
    if not conflicts:
        print("No conflicts found!")
        return False
    
    # Resolve each conflict region individually
    resolved_content = conflicted_content
    
    for i, conflict in enumerate(conflicts):
        resolution = resolve_conflict_with_ai(
            client, conflict, filepath, base_branch, i + 1, len(conflicts)
        )
        conflict['resolved'] = resolution
    
    # Replace all conflicts with resolutions (work backwards to maintain indices)
    for i in range(len(conflicts) - 1, -1, -1):
        conflict = conflicts[i]
        # Ensure resolved content ends with newline if the conflict region did
        resolution = conflict['resolved']
        if not resolution.endswith('\n'):
            resolution += '\n'
        resolved_content = resolved_content[:conflict['start']] + resolution + resolved_content[conflict['end']:]
    
    print(f"\n📊 Result:")
    print(f"   Original file: {len(conflicted_content.split(chr(10)))} lines")
    print(f"   Resolved file: {len(resolved_content.split(chr(10)))} lines")
    
    # Write the resolved content
    print(f"\n💾 Writing resolved content to {filepath}")
    with open(filepath, 'w') as f:
        f.write(resolved_content)
    
    # Stage the resolved file
    subprocess.run(['git', 'add', filepath], check=True)
    
    print(f"\n✅ Resolved: {filepath}")
    return True


def main():
    """Main entry point"""
    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Read conflicting files list
    conflicting_files_path = sys.argv[1] if len(sys.argv) > 1 else 'conflicting_files.txt'
    
    with open(conflicting_files_path) as f:
        conflicting_files = [line.strip() for line in f if line.strip()]
    
    print(f"Resolving conflicts in {len(conflicting_files)} files...")
    
    base_branch = os.environ.get('BASE_BRANCH', 'unknown')
    
    # Resolve each file
    success_count = 0
    for filepath in conflicting_files:
        try:
            if resolve_file_conflicts(client, filepath, base_branch):
                success_count += 1
        except Exception as e:
            print(f"\n❌ ERROR resolving {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Resolved {success_count}/{len(conflicting_files)} files!")
    print("="*60)
    
    if success_count < len(conflicting_files):
        sys.exit(1)


if __name__ == '__main__':
    main()


