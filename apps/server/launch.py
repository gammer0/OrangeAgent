"""``orange serve`` 的实际启动实现。

核心层（``src/orange``）禁止导入任何 Web 框架，因此把 uvicorn 与 ASGI
入口字符串都收在这里。``orange.cli._cmd_serve`` 只负责转发参数。

这个模块被设计成"薄"的：它唯一的职责是把 ASGI 应用跑起来，并把三类
常见启动失败翻译成一句人话——端口占用、工作目录不对、缺依赖。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ASGI 入口。uvicorn 的 reload 模式需要一个可导入的字符串，
#: 因此这里不能用对象引用。
ASGI_APP = "apps.server.main:app"

#: 端口占用在不同平台上的 errno：macOS 48、Linux 98、Windows 10048。
_PORT_IN_USE_ERRNOS = frozenset({48, 98, 10048})


def serve(
    *,
    workdir: str | Path,
    host: str,
    port: int,
    reload: bool = False,
    quiet: bool = False,
) -> int:
    """启动 Web 控制台，返回进程退出码。

    Args:
        workdir: 工作目录。``create_app`` 会从这里读 ``.env`` 并定位工具根，
            所以必须真的 ``chdir`` 过去，只传路径不够。
        host: 绑定地址。
        port: 绑定端口。
        reload: 是否开启 uvicorn 的自动重载（开发用）。
        quiet: 为真时不打印启动横幅（``--json`` 模式用）。

    Returns:
        0 表示正常退出（含 Ctrl+C），1 表示可预期的启动失败。
    """
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn 是声明依赖
        print("缺少 uvicorn，请重新安装：uv tool install . --force", file=sys.stderr)
        return 1

    resolved = Path(workdir).resolve()
    os.chdir(resolved)

    if not quiet:
        print(f"Orange 控制台：http://{host}:{port}")
        print(f"工作目录：{resolved}")
        print("按 Ctrl+C 停止\n")

    try:
        uvicorn.run(
            ASGI_APP,
            host=host,
            port=port,
            reload=bool(reload),
            log_level="info",
        )
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        print("\n已停止")
    except OSError as exc:
        # 端口占用是最常见的启动失败。原始 traceback 会盖住用户唯一
        # 需要知道的那件事：换个端口。
        if getattr(exc, "errno", None) in _PORT_IN_USE_ERRNOS or "10048" in str(exc):
            print(
                f"端口 {port} 已被占用。换一个端口：orange serve --port {port + 1}",
                file=sys.stderr,
            )
            return 1
        raise
    return 0
