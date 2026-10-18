只做一个动作：

Set-Location C:\\Users\\USER\\Desktop\\agent; Remove-Item -Force .tmp_snake_logic.py; Test-Path .tmp_snake_logic.py

目标是输出 False。不要读文件、不要分析、不要做其他事。
