from pathlib import Path

import sympy as sp
from sympy.printing.numpy import NumPyPrinter


def save_functions(
    path: str | Path,
    matrices: dict[str, sp.MatrixBase],
    q: sp.MatrixBase,
    qd: sp.MatrixBase,
) -> None:
    """Save symbolic matrices as functions returning NumPy arrays.

    Each generated function takes one state vector ``x`` containing ``q``
    followed by ``qd``, then the same parameter list. Dynamic symbols such as
    ``theta1(t)`` and their derivatives become local names ``theta1`` and
    ``theta1d``.
    """
    path = Path(path)
    printer = NumPyPrinter()

    q_items = [row[0] for row in q.tolist()]
    qd_items = [row[0] for row in qd.tolist()]
    q_symbols = [sp.Symbol(str(item).split("(", 1)[0]) for item in q_items]
    qd_symbols = [sp.Symbol(f"{symbol}d") for symbol in q_symbols]
    replacements = {
        **dict(zip(q_items, q_symbols)),
        **dict(zip(qd_items, qd_symbols)),
    }

    plain_matrices = {
        name: matrix.applyfunc(lambda expr: expr.xreplace(replacements))
        for name, matrix in matrices.items()
    }

    all_state_symbols = q_symbols + qd_symbols
    all_matrix_symbols = set()
    for matrix in plain_matrices.values():
        all_matrix_symbols.update(matrix.free_symbols)

    parameter_symbols = sorted(
        all_matrix_symbols.difference(all_state_symbols),
        key=str,
    )

    def expression_code(expr: sp.Expr) -> str:
        return str(printer.doprint(expr))

    def matrix_code(matrix: sp.MatrixBase) -> str:
        rows = []
        for row in matrix.tolist():
            entries = ",\n        ".join(
                expression_code(expr)
                for expr in row
            )
            rows.append(f"        [{entries}]")
        return "numpy.array([\n" + ",\n".join(rows) + "\n    ], dtype=float)"

    lines = [
        '"""Generated numerical dynamics functions."""',
        "import numpy",
        "",
    ]

    state_unpack = ", ".join(str(symbol) for symbol in all_state_symbols)
    parameter_names = ", ".join(str(symbol) for symbol in parameter_symbols)
    function_arguments = ", ".join(
        argument
        for argument in ("x", parameter_names)
        if argument
    )

    for name, matrix in plain_matrices.items():

        lines.extend([
            f"def calculate_{name}({function_arguments}):",
            "    x = numpy.asarray(x).flatten()",
            f"    [{state_unpack}] = x",
            f"    {name} = {matrix_code(matrix)}",
            f"    return {name}.squeeze()",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")
