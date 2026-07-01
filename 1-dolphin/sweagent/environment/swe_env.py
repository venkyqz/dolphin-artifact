import asyncio
import logging
import shlex
from pathlib import PurePath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    DockerMountDeploymentConfig,
    DockerStandbyDeploymentConfig,
    LocalDeploymentConfig,
    RemoteDeploymentConfig,
    WorkspaceDeploymentConfig,
    get_deployment,
)
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    CreateBashSessionRequest,
    ReadFileRequest,
    WriteFileRequest,
)
from swerex.runtime.abstract import Command as RexCommand
from swerex.runtime.abstract import CommandResponse as RexCommandResponse

from sweagent.environment.hooks.abstract import CombinedEnvHooks, EnvHook
from sweagent.environment.repo import PreExistingRepoConfig, Repo, RepoConfig
from sweagent.utils.log import get_logger


class EnvironmentConfig(BaseModel):
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11", python_standalone_dir="/root"),
        description="Deployment options.",
    )
    repo: RepoConfig | None = Field(
        default=None,
        description="Repository options.",
    )
    post_startup_commands: list[str] = []
    """Execute these commands before starting to run the agent but after all other setup steps.
    They will be executed in the same shell as the agent.
    Note: Every command is passed as a string, not a list of arguments.
    """
    post_startup_command_timeout: int = 500
    """Timeout for the post-startup commands.
    NOTE: The timeout applies to every command in `post_startup_commands` separately.
    """

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    name: str = "main"


class SWEEnv:
    def __init__(
        self,
        *,
        deployment: AbstractDeployment,
        repo: Repo | RepoConfig | None,
        post_startup_commands: list[str],
        post_startup_command_timeout: int = 500,
        hooks: list[EnvHook] | None = None,
        name: str = "main",
        local_root: str = "",
        remote_root: str = "",
    ):
        """This class represents the environment in which we solve the tasks.

        Args:
            deployment: SWE-ReX deployment instance
            repo: Repository configuration object, or anything following the `Repo` protocol
            post_startup_commands: Commands to execute before starting the agent
            hooks: Environment hooks (used to inject custom functionality)
                Equivalent to calling `add_hook` for each hook after initialization.
            name: Name of the environment
            local_root: Root of the workspace directory to use locally, used to persist information, this could be
                empty for example in the native docker deployment we do not have local workspace.
            remote_root: the directory where the code exists for executing bach-like commands such as editor,
                for example, in docker deployment, the default is "/", in local deployment, the default is still "/",
                but we could change it to others, in docker mount deployment, the remote_root is the directory
                inside the docker, and the local_root is the directory in local computer that will be
                mounted to docker.
        """
        super().__init__()
        self.deployment = deployment
        self.repo = repo
        self._post_startup_commands = post_startup_commands
        self.post_startup_command_timeout = post_startup_command_timeout
        self.logger = get_logger("swea-env", emoji="🪴")
        self.name = name
        self.local_root = local_root
        self.remote_root = remote_root
        self.clean_multi_line_functions = lambda x: x
        self._chook = CombinedEnvHooks()
        for hook in hooks or []:
            self.add_hook(hook)

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> Self:
        """Create an environment instance from a configuration object.
        This is the recommended way to create an environment instance, unless you need
        more flexibility.
        """
        # Always copy config to avoid shared state between different instances
        config = config.model_copy(deep=True)

        # we add a local workspace for non-bash inspection of local code copy, do not use docker
        local_root: str = ""
        remote_root: str = ""
        if isinstance(config.deployment, WorkspaceDeploymentConfig):
            # workspace deployment does not execution environment, we should copy repo to workspace
            # but still do not execute other commands
            local_root = config.deployment.workspace_root
        elif isinstance(config.deployment, DockerMountDeploymentConfig) or isinstance(
            config.deployment, DockerStandbyDeploymentConfig
        ):
            # all bash operations are done inside the docker, so the remote_root is the remote path
            local_root = config.deployment.local_mount_path
            remote_root = config.deployment.remote_mount_path
        elif isinstance(config.deployment, RemoteDeploymentConfig):
            remote_root = config.deployment.remote_root
            local_root = config.deployment.local_root
        elif isinstance(config.deployment, LocalDeploymentConfig):
            remote_root = config.deployment.path
            local_root = config.deployment.path
        else:
            # for other swe native deployment, the default execution path is "/"
            remote_root = "/"

        return cls(
            deployment=get_deployment(config.deployment),
            repo=config.repo,
            post_startup_commands=config.post_startup_commands,
            post_startup_command_timeout=config.post_startup_command_timeout,
            name=config.name,
            local_root=local_root,
            remote_root=remote_root,
        )

    def add_hook(self, hook: EnvHook) -> None:
        """Add `EnvHook` to the environment.

        This allows to inject custom functionality at different stages of the environment
        lifecycle, in particular to connect SWE-agent to a new interface (like a GUI).
        """
        hook.on_init(env=self)
        self._chook.add_hook(hook)

    def start(self) -> None:
        """Start the environment and reset it to a clean state."""
        self._init_deployment()
        self.reset()
        for command in self._post_startup_commands:
            self.communicate(command, check="raise", timeout=self.post_startup_command_timeout)

    def _copy_repo(self) -> None:
        """Clone/copy repository/codebase in container"""
        if self.repo is None:
            return

        # if folders have multiple files, it could have \t
        # if we mount dir and preexistingrepo, we should perform copy to overwrite exist dirs
        if not isinstance(self.repo, PreExistingRepoConfig):
            folders = self.communicate(input="ls", check="raise").split()
            if self.repo.repo_name in folders:
                return

        self._chook.on_copy_repo_started(repo=self.repo)
        self.repo.copy(self.deployment, self.remote_root)

    def hard_reset(self):
        """Resets the environment and deployment, i.e., completely restarts the
        deployment.
        """
        self.close()
        self.start()

    def get_repo_clone_path(self) -> str:
        """The path where the repo code should be cloned to and reset."""

        path: str = ""
        if self.local_root != "" and self.remote_root == "":
            # if no remote_root, it is workspace deployment, we should copy code to local_workspace_root instead
            return self.local_root
        elif self.remote_root != "":
            return self.remote_root
        else:
            raise RuntimeError("No local_root and no remote_root, please check your configuration.")

    def get_cwd(self) -> str:
        if self.repo is None:
            repo_dir = f"{self.get_repo_clone_path()}"
        else:
            repo_dir = f"{self.get_repo_clone_path()}/{self.repo.repo_name}"

        return repo_dir

    def reset(self):
        """Reset the environment to a clean state.
        Gets called by `start`, but can also be called independently to reset the
        environment to a clean state before a new attempt.

        Returns:
            observation: output from container
            info: additional information (e.g. debugging information)
        """

        import sys

        if sys.platform == "win32":
            # Use cmd.exe compatible cd command
            self.communicate(input=f"cd /d {self.get_repo_clone_path()}", check="raise")
        else:
            # Use bash cd command for Unix-like systems
            self.communicate(input=f"cd {self.get_repo_clone_path()}", check="raise")

        # if we use GithubRepoConfig, the _copy_repo will clone repo to the current
        # working dir, which is "/" inside docker, and _reset_repository by default will
        # cd to "/repo", if we want to copy github_url to local workspace, we should modify _copy_repo
        self._copy_repo()
        self._reset_repository()
        self._chook.on_environment_startup()

    def _reset_repository(self) -> None:
        """Clean repository of any modifications + Checkout base commit"""
        if self.repo is not None:
            # Do not reset for pre-existing repositories to support persistence
            if isinstance(self.repo, PreExistingRepoConfig):
                return
            self.logger.debug("Resetting repository %s to commit %s", self.repo.repo_name, self.repo.base_commit)
            # todo: Currently has swe-ft specific change: The original repo.copy isn't called, because the repo is already
            # present. However, reset --hard <BRANCH> also doesn't work. So modified it here to do a checkout instead.

            import sys

            if sys.platform == "win32":
                # Use cmd.exe compatible commands
                startup_commands = [
                    f"cd /d {self.get_cwd()}",
                    "git status",
                    "git fetch",
                    f"git reset --hard {self.repo.base_commit}",
                    "git clean -fdq",
                ]
            else:
                # Use bash commands for Unix-like systems
                startup_commands = [
                    f"cd {self.get_cwd()}",
                    "export ROOT=$(pwd -P)",
                    "git status",
                    "git fetch",
                    f"git reset --hard {self.repo.base_commit}",
                    "git clean -fdq",
                ]

            self.communicate(
                input=" && ".join(startup_commands),
                check="raise",
                error_msg="Failed to clean repository",
                # Sometimes this is slow because it rebuilds some index
                timeout=120,
            )

    def reset_cwd(self) -> str:
        """Some agent stage might cd to other directory, i.e., reproducer, making the
        working_dir points to other dir, so we cannot retrieve correct state,
        return the dir path"""
        repo_dir = self.get_cwd()

        import sys

        if sys.platform == "win32":
            # Use cmd.exe compatible cd command
            self.communicate(f"cd /d {repo_dir}")
        else:
            # Use bash cd command for Unix-like systems
            self.communicate(f"cd {repo_dir}")
        return repo_dir

    def close(self) -> None:
        """Shutdown SWE-ReX deployment etc."""
        self.logger.info("Beginning environment shutdown...")
        asyncio.run(self.deployment.stop())
        self._chook.on_close()

    # MARK: Helper functions #

    def _init_deployment(
        self,
    ) -> None:
        """Handles container initialization. Defines container name and creates it.
        If cached_image is provided, it will use that image name instead of the default.
        """
        self._chook.on_start_deployment()
        asyncio.run(self.deployment.start(), debug=True)

        # Use appropriate startup sources based on platform
        import sys

        if sys.platform == "win32":
            # For WSL, use the user's bashrc if it exists
            startup_sources = ["~/.bashrc"]
        else:
            # For Unix-like systems, use the standard root bashrc
            startup_sources = ["/root/.bashrc"]

        asyncio.run(self.deployment.runtime.create_session(CreateBashSessionRequest(startup_source=startup_sources)))

        # Only set Unix-specific environment variables on Unix-like systems
        if sys.platform != "win32":
            self.set_env_variables({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})

        self.logger.info("Environment Initialized")

    def interrupt_session(self):
        self.logger.info("Interrupting session")
        asyncio.run(self.deployment.runtime.run_in_session(BashInterruptAction()))

    # todo: return exit code?
    def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Executes a command in the running shell. The details of this are handled by
        the SWE-ReX deployment/runtime.

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            check: `ignore`: do not extract exit code (more stable), `warn`: extract exit code and log error if
                exit code is non-zero, `raise`: raise error if exit code is non-zero
            error_msg: error message to raise if the command fails

        Returns:
            output: output from container. Note that the output typically strips leading
                and trailing whitespace when returned from the subprocess.
        """
        self.logger.log(logging.TRACE, "Input:\n%s", input)  # type: ignore
        # Map check parameter to rex_check parameter
        if check == "ignore":
            rex_check = "ignore"
        elif check == "warn":
            rex_check = "silent"  # Get exit code but don't raise on failure
        elif check == "raise":
            rex_check = "raise"  # Get exit code and raise on failure
        else:
            rex_check = "ignore"  # Default fallback

        r = asyncio.run(
            self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        )
        output = r.output
        self.logger.log(logging.TRACE, "Output:\n%s", output)  # type: ignore
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            self.logger.error(msg)
            if check == "raise":
                self.close()
                raise RuntimeError(msg)
        return output

    def read_file(
        self, path: str | PurePath, binary: bool = False, encoding: str | None = None, errors: str | None = None
    ) -> str | bytes:
        """Read file contents from container

        Args:
            path: Absolute path to file
            binary: read as binary or text
            encoding: Encoding to use when reading the file. None means default encoding.
                This is the same as the `encoding` argument of `Path.read_text()`
            errors: Error handling to use when reading the file. None means default error handling.
                This is the same as the `errors` argument of `Path.read_text()`

        Returns:
            file_contents: Contents of file as string
        """
        r = asyncio.run(
            self.deployment.runtime.read_file(
                ReadFileRequest(path=str(path), binary=binary, encoding=encoding, errors=errors)
            )
        )
        return r.content

    def write_file(self, path: str | PurePath, content: str) -> None:
        """Write content to file in container"""
        asyncio.run(self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content)))

    def set_env_variables(self, env_variables: dict[str, str]) -> None:
        """Set environment variables in the environment."""
        if not env_variables:
            self.logger.debug("No environment variables to set")
            return

        import sys

        if sys.platform == "win32":
            # Use cmd.exe syntax for setting environment variables
            _env_setters = [f"set {k}={v}" for k, v in env_variables.items()]
            command = " && ".join(_env_setters)
        else:
            # Use bash syntax for Unix-like systems
            _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
            command = " && ".join(_env_setters)

        self.communicate(command, check="raise")

    def execute_command(
        self,
        command: str | list[str],
        shell: bool = True,
        check: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> RexCommandResponse:
        """Execute a command in the environment independent of the session (i.e., as a subprocess)

        Args:
            command: Command to execute (string or list of strings)
            shell: Whether to execute the command through the shell
            check: Whether to check the exit code and raise on failure
            env: Environment variables to set for the command
            cwd: Working directory to execute the command in

        Returns:
            RexCommandResponse: Response object containing output and exit code.
                Note that the output in the response has not been stripped of
                leading or trailing whitespace.
        """
        if not cwd:
            cwd = self.get_cwd()
        return asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=shell, check=check, env=env, cwd=cwd))
        )
