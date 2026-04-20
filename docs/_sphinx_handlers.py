"""Sphinx event handlers and transforms for autodoc customization."""

import inspect
import re

from docutils import nodes
from sphinx.addnodes import pending_xref
from sphinx.transforms import SphinxTransform

# Mapping for creating xref nodes: pattern -> (role, display_name, reftarget)
TYPE_XREF_MAPPING = [
    (
        r"_HHWLike",
        "class",
        "HomogeneousHarmonicWaveform",
        "typed_lisa_toolkit.types.HomogeneousHarmonicWaveform",
    ),
    (
        r"_HWLike",
        "class",
        "HarmonicWaveform",
        "typed_lisa_toolkit.types.HarmonicWaveform",
    ),
    (
        r"_HHPWLike",
        "class",
        "HomogeneousHarmonicProjectedWaveform",
        "typed_lisa_toolkit.types.HomogeneousHarmonicProjectedWaveform",
    ),
    (
        r"_HPWLike",
        "class",
        "HarmonicProjectedWaveform",
        "typed_lisa_toolkit.types.HarmonicProjectedWaveform",
    ),
    (
        r"_PWLike",
        "class",
        "ProjectedWaveform",
        "typed_lisa_toolkit.types.ProjectedWaveform",
    ),
]

# Derive simple name mapping from xref patterns
# (for str.replace() in signatures/docstrings)
TYPE_NAME_MAPPING = {
    pattern.replace(r"\.", ".").replace(r"\b", ""): display_name
    for pattern, _, display_name, _ in TYPE_XREF_MAPPING
}


def _format_type_expr(tp: object) -> str:
    """Format a type expression for display in generic parameter bounds."""
    if isinstance(tp, str):
        text = tp.strip()
    elif hasattr(tp, "__forward_arg__"):
        text = str(tp.__forward_arg__).strip()
    else:
        name = getattr(tp, "__name__", None)
        text = name.strip() if isinstance(name, str) else str(tp).replace("typing.", "")

    return text


def _get_type_parameters_from_obj(obj: object) -> list[str]:
    """Extract class type parameters from runtime metadata when available."""
    params = (
        getattr(obj, "__type_params__", None)
        or getattr(obj, "type_params", None)
        or getattr(obj, "__parameters__", None)
    )
    if not params:
        return []

    entries: list[str] = []
    for param in params:
        pname = getattr(param, "__name__", None)
        name = pname if isinstance(pname, str) and pname else str(param)

        bound = getattr(param, "__bound__", None)
        if bound is not None:
            entries.append(f"{name}: {_format_type_expr(bound)}")
            continue

        constraints = getattr(param, "__constraints__", None)
        if constraints:
            ctext = " | ".join(_format_type_expr(c) for c in constraints)
            entries.append(f"{name}: {ctext}")
            continue

        entries.append(name)

    return entries


def has_numpy_parameters_section(doc: str) -> bool:
    """Check if docstring has a NumPy-style Parameters section."""
    lines = [line.rstrip() for line in doc.splitlines()]
    for i in range(len(lines) - 1):
        if lines[i].strip().lower() != "parameters":
            continue
        underline = lines[i + 1].strip()
        if underline and set(underline) == {"-"}:
            return True
    return False


def has_numpy_returns_section(doc: str) -> bool:
    """Check if docstring has a NumPy-style Returns section."""
    lines = [line.rstrip() for line in doc.splitlines()]
    for i in range(len(lines) - 1):
        if lines[i].strip().lower() != "returns":
            continue
        underline = lines[i + 1].strip()
        if underline and set(underline) == {"-"}:
            return True
    return False


def process_autodoc_signature(
    app,
    what,
    name,
    obj,
    options,
    signature,
    return_annotation,
):
    """Conditionally strip type annotations and transform private types in signatures."""  # noqa: E501
    del app, name, options  # Unused
    if signature is None or what not in {"function", "method", "class"}:
        return None

    doc = inspect.getdoc(obj) or ""

    # For classes with generic metadata, suppress __init__ details and expose
    # generic params in PEP-695-like form: ClassName[T: Bound].
    if what == "class":
        type_params = _get_type_parameters_from_obj(obj)
        if type_params:
            return f"[{', '.join(type_params)}]", None
        return "", None

    # Transform private type names to public types in the signature
    sig_transformed = signature
    for private_type, public_type in TYPE_NAME_MAPPING.items():
        sig_transformed = sig_transformed.replace(private_type, public_type)

    _to_return = None
    if has_numpy_parameters_section(doc):
        try:
            sig = inspect.signature(obj)
            params = [
                p.replace(annotation=inspect.Signature.empty)
                for p in sig.parameters.values()
            ]
            sig_no_hints = sig.replace(
                parameters=params,
                return_annotation=inspect.Signature.empty,
            )
            sig_no_hints_str = str(sig_no_hints)
            # Also transform in the stripped signature
            for private_type, public_type in TYPE_NAME_MAPPING.items():
                sig_no_hints_str = sig_no_hints_str.replace(private_type, public_type)
            _to_return = sig_no_hints_str, return_annotation
        except (TypeError, ValueError):
            # Fall back to Sphinx's computed signature if introspection fails.
            _to_return = sig_transformed, return_annotation
    else:
        # No Parameters section - just transform the signature
        _to_return = sig_transformed, return_annotation

    if has_numpy_returns_section(doc):
        _to_return = _to_return[0], None

    return _to_return


def process_autodoc_docstring(app, what, name, obj, options, lines):
    """Transform private type names to public types in generated docstrings."""
    del app, name, obj, options  # Unused
    if what not in {"function", "method", "class"}:
        return

    # Transform all private type names in all lines
    for i, _ in enumerate(lines):
        for private_type, public_type in TYPE_NAME_MAPPING.items():
            lines[i] = lines[i].replace(private_type, public_type)


class TransformPrivateTypes(SphinxTransform):
    """Transform private type names to reference nodes for public types in the doctree."""  # noqa: E501

    default_priority = 0  # Run very early so xref resolution can process our nodes

    def apply(self, **kwargs):
        """Walk the doctree and replace private type names with reference nodes."""
        del kwargs  # Unused
        for node in list(self.document.traverse(nodes.Text)):
            text = node.astext()

            # Collect all matches from all patterns with their positions
            matches = []
            for pattern, role, display_name, reftarget in TYPE_XREF_MAPPING:
                for match in re.finditer(pattern, text):
                    matches.append(
                        (
                            match.start(),
                            match.end(),
                            pattern,
                            role,
                            display_name,
                            reftarget,
                        ),
                    )

            if not matches:
                continue

            # Sort by start position and remove overlaps
            matches.sort(key=lambda m: m[0])
            filtered_matches = []
            last_end = 0
            for start, end, pattern, role, display_name, reftarget in matches:
                if start >= last_end:  # No overlap
                    filtered_matches.append(
                        (start, end, pattern, role, display_name, reftarget),
                    )
                    last_end = end

            if not filtered_matches:
                continue

            # Build new nodes list with pending_xref for proper Sphinx resolution
            new_nodes = []
            last_end = 0

            for start, end, _, _, display_name, reftarget in filtered_matches:
                # Add text before the match
                if start > last_end:
                    new_nodes.append(nodes.Text(text[last_end:start]))

                # Create a pending_xref node that Sphinx will resolve to a link
                xref = pending_xref(
                    "",
                    nodes.Text(display_name),
                    refdomain="py",
                    reftype="class",
                    reftarget=reftarget,
                    modname=None,
                    classname=None,
                )
                new_nodes.append(xref)
                last_end = end

            # Add remaining text
            if last_end < len(text):
                new_nodes.append(nodes.Text(text[last_end:]))

            # Replace the original node with the new ones
            if new_nodes:
                parent = node.parent
                node_index = parent.index(node)
                parent.remove(node)
                for i, new_node in enumerate(new_nodes):
                    parent.insert(node_index + i, new_node)
