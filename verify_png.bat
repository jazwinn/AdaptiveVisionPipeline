@echo off
python -c "import os; path=r'C:\Users\ngjaz\AppData\Roaming\Claude\local-agent-mode-sessions\d08c12ba-7a93-4ca0-8d10-b5b3c8090aec\02b8d970-102f-4cac-8d72-28e2ef8ac78c\agent\local_ditto_02b8d970-102f-4cac-8d72-28e2ef8ac78c\outputs\gui_screenshot.png'; print('EXISTS - size:', os.path.getsize(path)) if os.path.exists(path) else print('NOT FOUND')"
pause
