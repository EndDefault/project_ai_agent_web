"""LLM이 전달한 Python 코드를 검사하고 새 파일로 저장합니다."""

import re
from pathlib import Path

from langchain.tools import tool
from tool.timing import measure_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "generated"


@tool
@measure_time
def write_python_file(file_name: str, code: str) -> str:
    """Python 코드를 generated 폴더에 새 파일로 저장합니다. 실행하지 않습니다.
    file_name은 calculator.py처럼 경로 없는 파일명입니다.
    영문자 또는 밑줄로 시작하고 영문자, 숫자, 밑줄과 .py 확장자를 사용하세요.
    code에는 Markdown 코드 블록 표시 없이 실제 Python 코드와 줄바꿈을 전달하세요.
    문법이 잘못됐거나 같은 파일이 이미 존재하면 저장하지 않고 오류를 반환합니다.
    """
    # 간단한 모듈 파일명만 허용해 폴더 밖 쓰기와 경로 혼동을 방지합니다.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", file_name):
        return "오류: 경로 없이 영문자·숫자·밑줄로 된 .py 파일명을 입력하세요. 첫 글자는 숫자일 수 없습니다."

    reserved = {"con", "prn", "aux", "nul"}
    reserved.update(f"com{i}" for i in range(1, 10))
    reserved.update(f"lpt{i}" for i in range(1, 10))
    if Path(file_name).stem.lower() in reserved:
        return "오류: 운영체제 예약 이름은 파일명으로 사용할 수 없습니다."
    if not code.strip():
        return "오류: 저장할 Python 코드가 비어 있습니다."

    try:
        # compile은 검사만 합니다. exec를 호출하지 않으므로 실행되지 않습니다.
        compile(code, file_name, "exec")
    except (SyntaxError, ValueError) as error:
        line = getattr(error, "lineno", None)
        message = getattr(error, "msg", str(error))
        return f"문법 검사 실패: {line or '?'}행, {message}. 파일을 저장하지 않았습니다."

    expected_dir = PROJECT_ROOT / "generated"
    if GENERATED_DIR.resolve() != expected_dir:
        return "오류: generated 폴더가 다른 위치로 연결되어 있어 저장하지 않습니다."

    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        target = GENERATED_DIR / file_name
        # x 모드는 파일이 이미 있으면 실패하므로 기존 파일을 덮어쓰지 않습니다.
        with target.open("x", encoding="utf-8", newline="\n") as output:
            output.write(code)
            if not code.endswith("\n"):
                output.write("\n")
    except FileExistsError:
        return f"오류: generated/{file_name}이 이미 있습니다. 다른 파일명을 사용하세요."
    except (OSError, UnicodeError):
        return "오류: 파일 저장에 실패했습니다. generated 폴더 상태를 확인하세요."

    return (
        f"저장 완료: generated/{file_name}\n"
        "Python 문법 검사 통과. 코드는 실행하지 않았으며 동작의 정확성은 별도 확인이 필요합니다."
    )
