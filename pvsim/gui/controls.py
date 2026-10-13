"""Reusable control widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QWidget


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class ParamForm(QWidget):
    """Editable form for a flat dict of parameters (a dataclass's `to_dict()`).

    Numbers are typed as text so any magnitude works (``220e-6``, ``40000``).
    The field types come from `defaults`; a default of ``None`` means an
    optional float (empty box = None).  `choices` maps a field to the allowed
    values, shown as a combo box (`labels` gives their display text).
    `changed` fires when the user finishes editing a field.
    """

    changed = Signal()

    def __init__(self, defaults: dict, choices: dict | None = None, labels: dict | None = None,
                 tips: dict | None = None, skip: tuple = ()):
        super().__init__()
        self.defaults = {k: v for k, v in defaults.items() if k not in skip}
        self.choices = choices or {}
        self.widgets: dict[str, QWidget] = {}
        lay = QFormLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        labels, tips = labels or {}, tips or {}
        for k, v in self.defaults.items():
            if k in self.choices:
                w = QComboBox()
                for c in self.choices[k]:
                    w.addItem(labels.get((k, c), _fmt(c)), c)
                w.currentIndexChanged.connect(self.changed)
            elif isinstance(v, bool):
                w = QCheckBox()
                w.toggled.connect(self.changed)
            else:
                w = QLineEdit()
                w.setAlignment(Qt.AlignRight)
                w.editingFinished.connect(self._edited)
            if k in tips:
                w.setToolTip(tips[k])
            self.widgets[k] = w
            lay.addRow(k, w)
        self.set_values(self.defaults)

    def _edited(self):
        w = self.sender()
        if w is not None and w.isModified():
            w.setModified(False)
            self.changed.emit()

    def values(self) -> dict:
        """Current values; raises ValueError naming the field that does not parse."""
        out = {}
        for k, default in self.defaults.items():
            w = self.widgets[k]
            if isinstance(w, QComboBox):
                out[k] = w.currentData()
            elif isinstance(w, QCheckBox):
                out[k] = w.isChecked()
            else:
                text = w.text().strip()
                try:
                    if default is None:
                        out[k] = float(text) if text else None
                    elif isinstance(default, int):
                        out[k] = int(text)
                    elif isinstance(default, float):
                        out[k] = float(text)
                    else:
                        out[k] = text
                except ValueError:
                    raise ValueError(f"{k}: {text!r} is not a valid number") from None
        return out

    def set_values(self, d: dict) -> None:
        """Show the values in `d` (unknown keys ignored) without emitting `changed`."""
        self.blockSignals(True)
        try:
            for k, v in d.items():
                w = self.widgets.get(k)
                if w is None:
                    continue
                if isinstance(w, QComboBox):
                    idx = next((i for i in range(w.count())
                                if _close(w.itemData(i), v)), -1)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                else:
                    w.setText(_fmt(v))
                    w.setModified(False)
        finally:
            self.blockSignals(False)


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            return False
    return a == b
