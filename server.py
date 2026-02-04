import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DEFAULT_TIMEOUT = 500.0
PKG_PATTERN = re.compile(r"\bPkg\.")
TEMP_SESSION_KEY = "__temp__"

mcp_server = FastMCP("julia")


class JuliaSession:
    def __init__(
        self,
        env_dir: str,
        *,
        is_temp: bool = False,
    ):
        self.env_dir = env_dir
        self.is_temp = is_temp
        self.tmux_session: str | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            raise RuntimeError(
                "Julia not found in PATH. Install from https://julialang.org/downloads/"
            )

        tmux = shutil.which("tmux")
        if tmux is None:
            raise RuntimeError(
                "tmux not found in PATH."
            )

        # Create unique tmux session name
        self.tmux_session = f"julia-mcp-{uuid.uuid4().hex[:8]}"

        julia_cmd = (
            f"{julia} -i "
            f"--project={self.env_dir}"
        )

        # Create a new tmux session running Julia (detached initially)
        create_cmd = [
            tmux, "new-session", "-d", "-s", self.tmux_session,
            "-c", self.env_dir,
            julia_cmd,
        ]
        proc = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"Failed to create tmux session: {stderr.decode()}")

        # Open a terminal window attached to this tmux session
        await self._open_terminal_window()

        # Wait for Julia to be ready
        await self._execute_raw(
            "",
            timeout=120.0,  # generous startup timeout
        )

        # Print a startup message indicating this is an MCP session
        await self._execute_raw(
            f'# Julia MCP session: {self.env_dir}',
            timeout=10.0,
        )

    async def _open_terminal_window(self) -> None:
        """Open a terminal window attached to the tmux session."""
        tmux = shutil.which("tmux")
        attach_cmd = f"{tmux} attach-session -t {self.tmux_session}"
        
        # Try $TERMINAL environment variable first
        env_terminal = os.environ.get("TERMINAL")
        if env_terminal and shutil.which(env_terminal):
            cmd = [env_terminal, "-e", attach_cmd]
        # Then try x-terminal-emulator (Debian/Ubuntu default)
        elif shutil.which("x-terminal-emulator"):
            cmd = ["x-terminal-emulator", "-e", attach_cmd]
        else:
            raise RuntimeError(
                "No terminal emulator found. Please set the TERMINAL environment variable "
                "to your preferred terminal emulator (e.g., export TERMINAL=gnome-terminal)"
            )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
        # Give the terminal a moment to start
        await asyncio.sleep(0.5)

    def is_alive(self) -> bool:
        if self.tmux_session is None:
            return False
        # Check if tmux session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.tmux_session],
            capture_output=True,
        )
        return result.returncode == 0

    async def execute(self, code: str, timeout: float | None) -> str:
        async with self.lock:
            if not self.is_alive():
                raise RuntimeError("Julia session has died unexpectedly")
            # Wrap multi-line code in begin...end block so it's treated as single expression
            num_lines = code.count("\n")
            if num_lines > 1:
                wrapped = "begin\n" + code + "\nend"
                if code.endswith(";"):
                    wrapped = wrapped + ";"
                lines_to_skip = num_lines + 3
            else:
                wrapped = code
                lines_to_skip = 1
            return await self._execute_raw(wrapped, timeout, lines_to_skip)

    async def _execute_raw(self, code: str, timeout: float | None, lines_to_skip: int = 1) -> str:
        assert self.tmux_session is not None
        tmux = shutil.which("tmux")

        proc = await asyncio.create_subprocess_exec(
            tmux, "send-keys", "-t", self.tmux_session, "C-l",
        )
        await proc.wait()

        proc = await asyncio.create_subprocess_exec(
            tmux, "send-keys", "-l", "-t", self.tmux_session, code,
        )
        await proc.wait()

        proc = await asyncio.create_subprocess_exec(
            tmux, "send-keys", "-t", self.tmux_session, "Enter",
        )
        await proc.wait()

        async def read_until_prompt() -> str:
            """Poll tmux pane content until a new julia> prompt appears."""
            last_content = None
            stable_count = 0
            
            while True:
                await asyncio.sleep(0.1)
                
                # Capture the pane content
                proc = await asyncio.create_subprocess_exec(
                    tmux, "capture-pane", "-t", self.tmux_session, "-p",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                content = stdout.decode(errors='replace')

                # Check if session is still alive
                if not self.is_alive():
                    raise RuntimeError(
                        f"Julia process died during execution.\n"
                        f"Last pane content:\n{content}"
                    )

                # Check if content has stabilized with a julia> prompt at the end
                lines = content.rstrip().split('\n')
                if lines and lines[-1].strip() == "julia>":
                    # Content ends with empty prompt, check if it's stable
                    if content == last_content:
                        stable_count += 1
                        if stable_count >= 2:  # Stable for 0.2s
                            # return output
                            if lines[0].startswith("julia>"):
                                # Great unlikely that any output is truncated.
                                return "\n".join(lines[lines_to_skip:-1])
                            else:
                                # Assume output is truncated
                                return "WARNING output truncated. Last few lines: \n" + "\n".join(lines[:-1])
                    else:
                        stable_count = 0
                else:
                    stable_count = 0
                
                last_content = content

        if timeout is not None:
            try:
                return await asyncio.wait_for(read_until_prompt(), timeout=timeout)
            except asyncio.TimeoutError:
                await self.kill()
                raise RuntimeError(
                    f"Execution timed out after {timeout}s. "
                    "Session killed; it will restart on next call."
                )
        else:
            return await read_until_prompt()

    async def kill(self) -> None:
        if self.tmux_session is not None:
            # Kill the tmux session
            proc = await asyncio.create_subprocess_exec(
                "tmux", "kill-session", "-t", self.tmux_session,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.wait()
            self.tmux_session = None
        if self.is_temp and os.path.isdir(self.env_dir):
            shutil.rmtree(self.env_dir, ignore_errors=True)


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, JuliaSession] = {}
        self._create_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _key(self, env_path: str | None) -> str:
        if env_path is None:
            return TEMP_SESSION_KEY
        return str(Path(env_path).resolve())

    async def get_or_create(self, env_path: str | None) -> JuliaSession:
        key = self._key(env_path)

        # Fast path
        if key in self._sessions and self._sessions[key].is_alive():
            return self._sessions[key]

        # Get per-key creation lock
        async with self._global_lock:
            if key not in self._create_locks:
                self._create_locks[key] = asyncio.Lock()
            create_lock = self._create_locks[key]

        async with create_lock:
            # Double-check
            if key in self._sessions and self._sessions[key].is_alive():
                return self._sessions[key]

            # Clean up dead session
            if key in self._sessions:
                await self._sessions[key].kill()
                del self._sessions[key]

            # Create new session
            is_temp = env_path is None
            if is_temp:
                env_dir = tempfile.mkdtemp(prefix="julia-mcp-")
            else:
                env_dir = str(Path(env_path).resolve())

            session = JuliaSession(
                env_dir, is_temp=is_temp,
            )
            await session.start()
            self._sessions[key] = session
            return session

    async def restart(self, env_path: str | None) -> None:
        key = self._key(env_path)
        if key in self._sessions:
            await self._sessions[key].kill()
            del self._sessions[key]

    def list_sessions(self) -> list[dict]:
        return [
            {
                "env_path": session.env_dir,
                "alive": session.is_alive(),
                "temp": session.is_temp,
            }
            for session in self._sessions.values()
        ]

    async def shutdown(self) -> None:
        for session in self._sessions.values():
            await session.kill()
        self._sessions.clear()


manager = SessionManager()


@mcp_server.tool()
async def julia_eval(
    code: str,
    env_path: str | None = None,
    timeout: float | None = None,
) -> str:
    """ALWAYS use this tool to run Julia code. NEVER run julia via command line.

    Persistent REPL session with state preserved between calls.
    Each env_path gets its own session, started lazily.

    Args:
        code: Julia code to evaluate.
        env_path: Julia project directory path. Omit for a temporary environment.
        timeout: Seconds (default: 500). Auto-disabled for Pkg operations.
    """
    if timeout is None:
        effective_timeout: float | None = (
            None if PKG_PATTERN.search(code) else DEFAULT_TIMEOUT
        )
    else:
        effective_timeout = timeout if timeout > 0 else None

    try:
        session = await manager.get_or_create(env_path)
        output = await session.execute(code, timeout=effective_timeout)
        return output if output else "(no output)"
    except RuntimeError as e:
        # Clean up dead session so next call starts fresh
        key = manager._key(env_path)
        if key in manager._sessions and not manager._sessions[key].is_alive():
            del manager._sessions[key]
        return f"Error: {e}"


@mcp_server.tool()
async def julia_restart(env_path: str | None = None) -> str:
    """Restart a Julia session, clearing all state.

    IMPORTANT: Restarting is slow and loses all session state. Very rarely needed.
    Revise.jl is loaded automatically in every session, so code changes to loaded packages are picked up without restarting.
    Only restart as a last resort when the session is truly broken, or code changes that Revise cannot fix.

    Args:
        env_path: Environment to restart. If omitted, restarts the temporary session.
    """
    await manager.restart(env_path)
    return "Session restarted. A fresh session will start on next julia_eval call."


@mcp_server.tool()
async def julia_list_sessions() -> str:
    """List all active Julia sessions and their environments."""
    sessions = manager.list_sessions()
    if not sessions:
        return "No active Julia sessions."
    lines = []
    for s in sessions:
        status = "alive" if s["alive"] else "dead"
        label = f"{s['env_path']} (temp)" if s["temp"] else s["env_path"]
        lines.append(f"  {label}: {status}")
    return "Active Julia sessions:\n" + "\n".join(lines)


def main():
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
