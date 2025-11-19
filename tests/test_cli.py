import pytest
from click.testing import CliRunner
from hf_lifecycle.cli import cli


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'HuggingFace Lifecycle Manager' in result.output

    def test_cli_version(self, runner):
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0

    def test_auth_help(self, runner):
        result = runner.invoke(cli, ['auth', '--help'])
        assert result.exit_code == 0
        assert 'Authentication commands' in result.output

    def test_repo_help(self, runner):
        result = runner.invoke(cli, ['repo', '--help'])
        assert result.exit_code == 0
        assert 'Repository management' in result.output

    def test_checkpoint_help(self, runner):
        result = runner.invoke(cli, ['checkpoint', '--help'])
        assert result.exit_code == 0
        assert 'Checkpoint management' in result.output

    def test_dataset_help(self, runner):
        result = runner.invoke(cli, ['dataset', '--help'])
        assert result.exit_code == 0
        assert 'Dataset management' in result.output

    def test_metadata_help(self, runner):
        result = runner.invoke(cli, ['metadata', '--help'])
        assert result.exit_code == 0
        assert 'Metadata tracking' in result.output

    def test_metadata_capture(self, runner, tmp_path):
        output_file = tmp_path / "metadata.json"
        result = runner.invoke(cli, ['metadata', 'capture', '-o', str(output_file)])
        
        # Should succeed
        assert result.exit_code == 0
        assert output_file.exists()
        assert 'Metadata saved' in result.output

    def test_metadata_show(self, runner, tmp_path):
        # First capture metadata
        output_file = tmp_path / "metadata.json"
        runner.invoke(cli, ['metadata', 'capture', '-o', str(output_file)])
        
        # Then show it
        result = runner.invoke(cli, ['metadata', 'show', str(output_file)])
        assert result.exit_code == 0
        assert 'EXPERIMENT METADATA SUMMARY' in result.output
