"""Smart context compression for the researcher agent."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSymbol:
    """A single extracted code symbol (function, class, method)."""

    name: str
    type: str  # "function", "class", "method"
    source: str
    file_path: str
    start_line: int
    end_line: int
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class CompressedContext:
    """Compressed code context containing only relevant symbols."""

    symbols: list[ExtractedSymbol] = field(default_factory=list)
    call_graph: dict[str, list[str]] = field(default_factory=dict)
    total_tokens_saved: int = 0
    original_token_count: int = 0
    compressed_token_count: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return 1 - (self.compressed_token_count / self.original_token_count)

    def to_prompt_context(self) -> str:
        """Format compressed context for LLM consumption."""
        sections = []
        for symbol in self.symbols:
            header = f"# {symbol.file_path} :: {symbol.name} ({symbol.type}, L{symbol.start_line}-{symbol.end_line})"
            sections.append(f"{header}\n{symbol.source}")
        return "\n\n".join(sections)


class ContextCompressor:
    """
    AST-based context compression for the researcher agent.

    # full file context wastes 60% of tokens on imports and unrelated functions

    Instead of passing entire files to the coder, we parse the AST and extract
    only the functions/classes that are relevant to the task. We also follow
    the call graph one level deep to include callers and callees.

    # AST-based extraction: give the coder only the function it needs to modify + its callers
    """

    def __init__(
        self,
        max_context_tokens: int = 16000,
        include_callers: bool = True,
        include_callees: bool = True,
        max_depth: int = 1,
    ):
        self.max_context_tokens = max_context_tokens
        self.include_callers = include_callers
        self.include_callees = include_callees
        self.max_depth = max_depth

    def compress(
        self,
        file_contents: dict[str, str],
        target_symbols: list[str],
        target_files: list[str] | None = None,
    ) -> CompressedContext:
        """
        Extract only relevant symbols from file contents.

        Args:
            file_contents: mapping of file path to source code
            target_symbols: function/class names we care about
            target_files: if provided, only look in these files
        """
        result = CompressedContext()
        original_total = sum(self._estimate_tokens(c) for c in file_contents.values())
        result.original_token_count = original_total

        # parse all files into ASTs
        parsed_files: dict[str, ast.Module] = {}
        source_lines: dict[str, list[str]] = {}

        for path, content in file_contents.items():
            if target_files and path not in target_files:
                continue
            try:
                parsed_files[path] = ast.parse(content)
                source_lines[path] = content.splitlines()
            except SyntaxError:
                logger.debug(f"failed to parse {path}, including raw")
                continue

        # extract target symbols
        all_symbols = self._extract_all_symbols(parsed_files, source_lines)
        symbol_map = {s.name: s for s in all_symbols}

        # build call graph
        call_graph = self._build_call_graph(parsed_files)
        result.call_graph = call_graph

        # collect relevant symbols: targets + their callers/callees
        relevant_names: set[str] = set(target_symbols)

        for name in list(relevant_names):
            if self.include_callees and name in call_graph:
                relevant_names.update(call_graph[name])
            if self.include_callers:
                for caller, callees in call_graph.items():
                    if name in callees:
                        relevant_names.add(caller)

        # collect symbols, respecting token budget
        token_budget = self.max_context_tokens
        for name in sorted(relevant_names):
            if name in symbol_map:
                symbol = symbol_map[name]
                tokens = self._estimate_tokens(symbol.source)
                if tokens <= token_budget:
                    result.symbols.append(symbol)
                    token_budget -= tokens

        result.compressed_token_count = self.max_context_tokens - token_budget
        result.total_tokens_saved = original_total - result.compressed_token_count
        return result

    def extract_symbol(
        self,
        source: str,
        symbol_name: str,
        file_path: str = "<unknown>",
    ) -> ExtractedSymbol | None:
        """Extract a single symbol from source code by name."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == symbol_name:
                    return self._node_to_symbol(node, lines, file_path, "function")
            elif isinstance(node, ast.ClassDef):
                if node.name == symbol_name:
                    return self._node_to_symbol(node, lines, file_path, "class")
                # check methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == symbol_name:
                            return self._node_to_symbol(item, lines, file_path, "method")

        return None

    def _extract_all_symbols(
        self,
        parsed_files: dict[str, ast.Module],
        source_lines: dict[str, list[str]],
    ) -> list[ExtractedSymbol]:
        """Extract all top-level symbols from parsed files."""
        symbols = []

        for path, tree in parsed_files.items():
            lines = source_lines[path]
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(self._node_to_symbol(node, lines, path, "function"))
                elif isinstance(node, ast.ClassDef):
                    symbols.append(self._node_to_symbol(node, lines, path, "class"))
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            symbols.append(
                                self._node_to_symbol(item, lines, path, "method")
                            )

        return symbols

    def _node_to_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        lines: list[str],
        file_path: str,
        symbol_type: str,
    ) -> ExtractedSymbol:
        """Convert an AST node to an ExtractedSymbol."""
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        source = "\n".join(lines[start:end])

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{ast.dump(dec)}")

        docstring = ast.get_docstring(node)

        return ExtractedSymbol(
            name=node.name,
            type=symbol_type,
            source=source,
            file_path=file_path,
            start_line=node.lineno,
            end_line=end,
            decorators=decorators,
            docstring=docstring,
        )

    def _build_call_graph(
        self,
        parsed_files: dict[str, ast.Module],
    ) -> dict[str, list[str]]:
        """Build a simple call graph from AST analysis."""
        graph: dict[str, list[str]] = {}

        for _path, tree in parsed_files.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls = self._get_calls_in_function(node)
                    graph[node.name] = calls

        return graph

    def _get_calls_in_function(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Get all function calls within a function body."""
        calls = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return list(set(calls))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars per token on average for code)."""
        return len(text) // 4


def compress_for_task(
    file_contents: dict[str, str],
    task_description: str,
    target_symbols: list[str] | None = None,
    max_tokens: int = 16000,
) -> CompressedContext:
    """
    Convenience function: compress file contents for a given task.

    If target_symbols not provided, attempts to infer them from the task description
    by looking for function/class names mentioned in the text.
    """
    compressor = ContextCompressor(max_context_tokens=max_tokens)

    if not target_symbols:
        # simple heuristic: find identifiers in task description that match symbols in code
        target_symbols = _infer_symbols_from_task(file_contents, task_description)

    return compressor.compress(file_contents, target_symbols)


def _infer_symbols_from_task(
    file_contents: dict[str, str],
    task_description: str,
) -> list[str]:
    """Infer relevant symbols by matching identifiers in task description to code."""
    # collect all defined names from files
    all_names: set[str] = set()
    for content in file_contents.values():
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                all_names.add(node.name)

    # find which names appear in the task description
    words = set(task_description.replace(".", " ").replace("(", " ").replace(")", " ").split())
    matches = all_names & words

    return list(matches) if matches else list(all_names)[:10]
