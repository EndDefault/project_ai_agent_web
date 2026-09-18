"""일반 Python 함수: 경로 준비, 줄 검색, AST 범위 확인, 코드 추출."""

from langchain.tools import tool

import ast
import os
from pathlib import Path

from ai_agent.tool.timing import measure_time

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".next"}


def collect_python_files(root: Path) -> tuple[Path, ...]:
    """조사 시작 시 한 번 실행. 제외 폴더는 처음부터 탐색하지 않습니다."""
    root = root.resolve()
    files = []

    def report_error(error):
        # 조용히 파일을 누락하지 않고 호출자에게 실패를 알림
        raise error

    for directory, dirs, names in os.walk(root, onerror=report_error):
        dirs[:] = sorted(
            name for name in dirs
            if name not in EXCLUDED_DIRS
            and not (Path(directory) / name).is_symlink()
            and (Path(directory) / name).resolve().is_relative_to(root)
        )
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix == ".py" and path.is_file():
                if path.resolve().is_relative_to(root):
                    files.append(path)

    return tuple(sorted(files))


def node_start(node: ast.AST) -> int:
    """함수의 데코레이터도 함수 코드에 포함합니다."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [item.lineno for item in decorators])


def enclosing_range(nodes: list[ast.AST], line_number: int):
    """가장 안쪽 함수 전체를 우선 선택. 함수 밖에서는 가장 작은 문장."""
    functions = []
    statements = []
    for node in nodes:
        if not isinstance(node, ast.stmt) or node.end_lineno is None:
            continue
        if node_start(node) <= line_number <= node.end_lineno:
            statements.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)

    candidates = functions or statements
    if not candidates:
        # 최상위 주석 등은 AST 문장이 없으므로 일치한 줄 자체 반환
        return line_number, line_number, "주석 또는 문장 밖의 줄"

    selected = min(
        candidates,
        key=lambda node: (node.end_lineno - node_start(node), -node.col_offset),
    )
    label = type(selected).__name__
    if isinstance(selected, (ast.FunctionDef, ast.AsyncFunctionDef)):
        label += f": {selected.name}"
    return node_start(selected), selected.end_lineno, label


def search_code_blocks(
    root: Path,
    files: tuple[Path, ...],
    keyword: str,
    file_path: str = "",
    max_results: int = 5,
) -> str:
    """준비된 경로 안에서 검색 → AST 범위 결정 → 원본 코드 반환."""
    if not keyword.strip():
        return "오류: 코드에 포함된 검색어를 입력해주세요."
    if not 1 <= max_results <= 20:
        return "오류: max_results는 1~20이어야 합니다."

    root = root.resolve()
    selected_files = files
    if file_path:
        target = (root / file_path).resolve()
        if not target.is_relative_to(root):
            return "오류: 프로젝트 내부 경로만 사용할 수 있습니다."
        selected_files = tuple(path for path in files if path.resolve() == target)
        if not selected_files:
            return "시작 시 준비한 Python 파일 목록에 해당 경로가 없습니다. 경로 또는 제외 폴더를 확인하세요."

    results = []
    warnings = []
    truncated = False
    for path in selected_files:
        relative = path.relative_to(root).as_posix()
        # 목록 준비 후 파일이 외부 링크로 바뀐 경우도 확인
        if not path.resolve().is_relative_to(root):
            warnings.append(f"{relative}: 외부 경로로 변경되어 건너뜀")
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            warnings.append(f"{relative}: 파일 읽기 실패")
            continue

        lines = content.splitlines()
        hit_lines = [number for number, line in enumerate(lines, 1) if keyword in line]
        if not hit_lines:
            continue

        try:
            nodes = list(ast.walk(ast.parse(content)))
        except SyntaxError as error:
            warnings.append(f"{relative}:{error.lineno}: 문법 오류. 이 파일은 일치한 줄만 반환합니다.")
            nodes = []

        # 동일 함수에 여러 줄이 일치해도 함수 코드는 한 번만 반환
        blocks = {}
        for number in hit_lines:
            start, end, label = enclosing_range(nodes, number)
            if (start, end) not in blocks:
                blocks[(start, end)] = {"label": label, "hits": []}
            blocks[(start, end)]["hits"].append(number)

        for (start, end), block in blocks.items():
            if len(results) >= max_results:
                truncated = True
                break
            numbered_code = "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(lines[start - 1:end], start)
            )
            hits = ", ".join(str(number) for number in block["hits"])
            results.append(
                f"파일: {relative}\n일치한 줄: {hits}\n"
                f"구조: {block['label']}\n범위: {start}~{end}행\n{numbered_code}"
            )
        if truncated:
            break

    output = "\n\n---\n\n".join(results) if results else "검색 결과가 없습니다."
    if truncated:
        output += f"\n\n결과를 {max_results}개 코드 범위로 제한했습니다. 파일 또는 검색어를 좁혀주세요."
    if warnings:
        output += "\n\n확인 사항:\n" + "\n".join(warnings)
    return output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def describe_statement(node: ast.stmt) -> str:
    """실제 값 대신 변수명과 호출 이름으로 최상위 문장을 짧게 표시합니다."""
    if isinstance(node, ast.Assign):
        targets = ", ".join(ast.unparse(target) for target in node.targets)
        label = f"대입: {targets}"
    elif isinstance(node, ast.AnnAssign):
        label = f"타입 지정 대입: {ast.unparse(node.target)}"
    else:
        label = type(node).__name__

    # 아래 블록 전체가 아니라 현재 문장의 표현식만 검사합니다.
    calls = []
    for field, value in ast.iter_fields(node):
        if field in {"body", "orelse", "finalbody", "handlers", "cases"}:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, ast.AST):
                continue
            for child in ast.walk(item):
                if isinstance(child, ast.Call) and isinstance(child.func, (ast.Name, ast.Attribute)):
                    name = ast.unparse(child.func)
                    if name not in calls:
                        calls.append(name)
    if calls:
        label += " / 호출: " + ", ".join(calls)
    return label


def outline_python_code(content: str, max_items: int) -> str:
    """전체 AST 덤프 대신 import·정의·최상위 실행 구조만 요약합니다."""
    tree = ast.parse(content)
    entries = []

    class OutlineVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit(self, node):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                scope = ".".join(self.scope) or "모듈"
                entries.append((node.lineno, node.end_lineno, f"import ({scope}): {ast.unparse(node)}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join(self.scope + [node.name])
                kind = "클래스" if isinstance(node, ast.ClassDef) else "함수"
                if isinstance(node, ast.AsyncFunctionDef):
                    kind = "비동기 함수"
                entries.append((node_start(node), node.end_lineno, f"{kind}: {name}"))
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()
                return
            elif isinstance(node, ast.stmt) and not self.scope:
                # 문서 문자열은 구조 요약에서 제외합니다.
                is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                if not is_doc:
                    entries.append((node.lineno, node.end_lineno, describe_statement(node)))
            self.generic_visit(node)

    OutlineVisitor().visit(tree)
    output = [f"전체 {len(content.splitlines())}행 / 구조 항목 {len(entries)}개"]
    for start, end, label in entries[:max_items]:
        output.append(f"{start}~{end}행 | {label}")
    if not entries:
        output.append("표시할 문장이 없습니다. 빈 파일 또는 주석만 있는 파일일 수 있습니다.")
    if len(entries) > max_items:
        output.append(f"처음 {max_items}개만 표시했습니다. max_items를 늘리거나 확인한 이름으로 검색하세요.")
    output.append("구조 요약입니다. 함수 본문이나 실제 동작은 표시된 이름으로 search_project_context를 호출해 확인하세요.")
    return "\n".join(output)


@tool
@measure_time
def inspect_project_file(file_path: str, max_items: int = 100) -> str:
    """검색어 없이 Python 파일의 구조를 확인합니다.
    file_path는 프로젝트 기준 상대 경로(예: agent_main.py)입니다.
    import, 함수·클래스 이름, 최상위 실행 문장과 실제 줄 범위를 반환합니다.
    검색어를 모를 때 먼저 사용한 뒤 발견한 이름으로 search_project_context를 호출하세요.
    파일을 실행하지 않으며 함수 본문 전체나 변수의 실제 값은 반환하지 않습니다.
    max_items는 표시할 구조 항목 수로 1~500입니다.
    """
    if not file_path.strip():
        return "오류: 파일의 상대 경로를 입력해주세요."
    if not 1 <= max_items <= 500:
        return "오류: max_items는 1~500이어야 합니다."
    path = (PROJECT_ROOT / file_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        return "오류: 프로젝트 내부 파일만 확인할 수 있습니다."
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return "오류: 검색 제외 폴더의 파일입니다."
    if not path.is_file() or path.suffix != ".py":
        return "오류: 존재하는 Python 파일 경로를 입력해주세요."
    try:
        content = path.read_text(encoding="utf-8-sig")
        outline = outline_python_code(content, max_items)
    except (OSError, UnicodeError):
        return "오류: 파일을 읽을 수 없습니다."
    except SyntaxError as error:
        return f"문법 오류로 구조를 분석할 수 없습니다: {error.lineno}행, {error.msg}"
    return f"파일: {relative.as_posix()}\n{outline}"


@tool
@measure_time
def search_project_context(
    keyword: str, file_path: str = "", max_results: int = 5
) -> str:
    """Python 코드에서 문자열을 검색하고 포함 함수 또는 문장의 실제 코드를 반환합니다.
    keyword는 코드에 포함될 문자열이며 대소문자를 구분합니다.
    file_path는 프로젝트 기준 상대 경로입니다. 생략하면 프로젝트의 Python 파일을 검색합니다.
    max_results는 반환할 코드 범위 수(1~20)입니다.
    파일 목록 준비, 검색, AST 범위 확인, 줄 번호를 붙인 코드 읽기를 자동 수행합니다.
    예: keyword='PostgresSaver', file_path='agent_main.py'
    """
    # LLM의 추가 선택 없이 Python이 먼저 최신 파일 목록을 준비합니다.
    try:
        project_files = collect_python_files(PROJECT_ROOT)
    except OSError:
        return "오류: 프로젝트 파일 목록을 준비하지 못했습니다. 접근 권한을 확인하세요."

    return search_code_blocks(
        PROJECT_ROOT, project_files, keyword, file_path, max_results
    )
