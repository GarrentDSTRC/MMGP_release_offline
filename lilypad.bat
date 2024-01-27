@echo off
echo Current path is %cd%
for /l %%i in (6,1,6) do (
 start "" processing-java --force --sketch=D:\SourceCode\MMGP_OL%%i --output=D:\SourceCode\MMGP_OL%%i\output --run
)
pause

