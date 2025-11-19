"""
Command Line Interface for HuggingFace Lifecycle Manager.
"""
import click
import sys
from pathlib import Path

from hf_lifecycle.auth import AuthManager
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.model_registry import ModelRegistry
from hf_lifecycle.dataset import DatasetManager
from hf_lifecycle.metadata import MetadataTracker
from hf_lifecycle import __version__


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """HuggingFace Lifecycle Manager - Comprehensive ML lifecycle management."""
    ctx.ensure_object(dict)


# ============================================================================
# AUTH COMMANDS
# ============================================================================
@cli.group()
def auth():
    """Authentication commands."""
    pass


@auth.command()
@click.option('--token', prompt=True, hide_input=True, help='HuggingFace token')
@click.option('--write-to-disk/--no-write-to-disk', default=True, help='Save token to disk')
def login(token, write_to_disk):
    """Login to HuggingFace Hub."""
    try:
        auth_mgr = AuthManager()
        auth_mgr.login(token=token, write_to_disk=write_to_disk)
        click.echo(click.style('✓ Successfully logged in!', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Login failed: {e}', fg='red'), err=True)
        sys.exit(1)


@auth.command()
def logout():
    """Logout from HuggingFace Hub."""
    try:
        auth_mgr = AuthManager()
        auth_mgr.logout()
        click.echo(click.style('✓ Successfully logged out!', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Logout failed: {e}', fg='red'), err=True)
        sys.exit(1)


@auth.command()
def whoami():
    """Show current user information."""
    try:
        auth_mgr = AuthManager()
        token = auth_mgr.get_token()
        if token:
            auth_mgr.validate_token()
            click.echo(click.style('✓ Token is valid', fg='green'))
        else:
            click.echo(click.style('✗ Not logged in', fg='yellow'))
    except Exception as e:
        click.echo(click.style(f'✗ Error: {e}', fg='red'), err=True)
        sys.exit(1)


# ============================================================================
# REPO COMMANDS
# ============================================================================
@cli.group()
def repo():
    """Repository management commands."""
    pass


@repo.command()
@click.argument('repo_id')
@click.option('--type', 'repo_type', type=click.Choice(['model', 'dataset', 'space']), default='model')
@click.option('--private/--public', default=True)
def create(repo_id, repo_type, private):
    """Create a new repository."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        url = repo_mgr.create_repo(repo_id, repo_type=repo_type, private=private)
        click.echo(click.style(f'✓ Created {repo_type} repository: {url}', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Failed to create repository: {e}', fg='red'), err=True)
        sys.exit(1)


@repo.command()
@click.argument('repo_id')
@click.option('--type', 'repo_type', type=click.Choice(['model', 'dataset', 'space']), default='model')
@click.confirmation_option(prompt='Are you sure you want to delete this repository?')
def delete(repo_id, repo_type):
    """Delete a repository."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        repo_mgr.delete_repo(repo_id, repo_type=repo_type)
        click.echo(click.style(f'✓ Deleted {repo_type} repository: {repo_id}', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Failed to delete repository: {e}', fg='red'), err=True)
        sys.exit(1)


@repo.command()
@click.option('--username', help='Username to list repos for (default: current user)')
def list(username):
    """List repositories."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        repos = repo_mgr.list_repos(username=username)
        
        if repos:
            click.echo(click.style(f'Found {len(repos)} repositories:', fg='cyan'))
            for repo in repos:
                click.echo(f'  • {repo}')
        else:
            click.echo(click.style('No repositories found', fg='yellow'))
    except Exception as e:
        click.echo(click.style(f'✗ Failed to list repositories: {e}', fg='red'), err=True)
        sys.exit(1)


# ============================================================================
# CHECKPOINT COMMANDS
# ============================================================================
@cli.group()
def checkpoint():
    """Checkpoint management commands."""
    pass


@checkpoint.command()
@click.option('--local-dir', default='./checkpoints', help='Local checkpoint directory')
def list_checkpoints(local_dir):
    """List available checkpoints."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        ckpt_mgr = CheckpointManager(repo_mgr, local_dir=local_dir)
        
        checkpoints = ckpt_mgr.list_checkpoints()
        if checkpoints:
            click.echo(click.style(f'Found {len(checkpoints)} checkpoints:', fg='cyan'))
            for ckpt in checkpoints:
                click.echo(f"  • {ckpt['name']} (step: {ckpt.get('step', 'N/A')})")
        else:
            click.echo(click.style('No checkpoints found', fg='yellow'))
    except Exception as e:
        click.echo(click.style(f'✗ Failed to list checkpoints: {e}', fg='red'), err=True)
        sys.exit(1)


@checkpoint.command()
@click.option('--local-dir', default='./checkpoints', help='Local checkpoint directory')
@click.option('--dry-run/--no-dry-run', default=False, help='Show what would be deleted')
def cleanup(local_dir, dry_run):
    """Cleanup old checkpoints based on retention policy."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        ckpt_mgr = CheckpointManager(repo_mgr, local_dir=local_dir)
        
        deleted = ckpt_mgr.cleanup(dry_run=dry_run)
        if dry_run:
            click.echo(click.style(f'Would delete {len(deleted)} checkpoints:', fg='yellow'))
        else:
            click.echo(click.style(f'✓ Deleted {len(deleted)} checkpoints:', fg='green'))
        
        for name in deleted:
            click.echo(f'  • {name}')
    except Exception as e:
        click.echo(click.style(f'✗ Cleanup failed: {e}', fg='red'), err=True)
        sys.exit(1)


# ============================================================================
# DATASET COMMANDS
# ============================================================================
@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command()
@click.argument('repo_id')
@click.option('--private/--public', default=True)
def create_dataset(repo_id, private):
    """Create a new dataset repository."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        dataset_mgr = DatasetManager(repo_mgr)
        
        url = dataset_mgr.create_dataset(repo_id, private=private)
        click.echo(click.style(f'✓ Created dataset: {url}', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Failed to create dataset: {e}', fg='red'), err=True)
        sys.exit(1)


@dataset.command()
@click.argument('repo_id')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--path-in-repo', help='Path in repository (default: same as filename)')
def upload(repo_id, file_path, path_in_repo):
    """Upload a file to a dataset."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        dataset_mgr = DatasetManager(repo_mgr)
        
        if path_in_repo is None:
            path_in_repo = Path(file_path).name
        
        dataset_mgr.upload_file(repo_id, file_path, path_in_repo)
        click.echo(click.style(f'✓ Uploaded {file_path} to {repo_id}', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Upload failed: {e}', fg='red'), err=True)
        sys.exit(1)


@dataset.command()
@click.argument('repo_id')
@click.argument('local_dir', type=click.Path())
def download(repo_id, local_dir):
    """Download a dataset."""
    try:
        auth_mgr = AuthManager()
        repo_mgr = RepoManager(auth_mgr)
        dataset_mgr = DatasetManager(repo_mgr)
        
        path = dataset_mgr.download_dataset(repo_id, local_dir)
        click.echo(click.style(f'✓ Downloaded to: {path}', fg='green'))
    except Exception as e:
        click.echo(click.style(f'✗ Download failed: {e}', fg='red'), err=True)
        sys.exit(1)


# ============================================================================
# METADATA COMMANDS
# ============================================================================
@cli.group()
def metadata():
    """Metadata tracking commands."""
    pass


@metadata.command()
@click.option('--output', '-o', type=click.Path(), default='metadata.json', help='Output file')
def capture(output):
    """Capture system and environment metadata."""
    try:
        tracker = MetadataTracker()
        tracker.capture_system_info()
        tracker.capture_environment()
        tracker.save_metadata(output)
        
        click.echo(click.style(f'✓ Metadata saved to: {output}', fg='green'))
        click.echo('\n' + tracker.get_summary())
    except Exception as e:
        click.echo(click.style(f'✗ Failed to capture metadata: {e}', fg='red'), err=True)
        sys.exit(1)


@metadata.command()
@click.argument('metadata_file', type=click.Path(exists=True))
def show(metadata_file):
    """Display metadata summary."""
    try:
        tracker = MetadataTracker()
        tracker.load_metadata(metadata_file)
        click.echo(tracker.get_summary())
    except Exception as e:
        click.echo(click.style(f'✗ Failed to load metadata: {e}', fg='red'), err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
