from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from SUPMT_score import predict_risk_score
from src.preprocessing.raw_input_mapping import InputMappingError
from SUPMT import SUPMTError


class RiskApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BCSC Risk Estimation")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        self.values: dict[str, tk.StringVar] = {}
        self._configure_style()
        self._build_form()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white", foreground="black")
        style.configure("TButton", background="white", foreground="black", borderwidth=1)
        style.configure("TCombobox", fieldbackground="white", background="white", foreground="black")
        style.configure("Result.TLabel", background="white", foreground="black", font=("Segoe UI", 22, "bold"))
        style.configure("Error.TLabel", background="white", foreground="black")

    def _build_form(self) -> None:
        frame = ttk.Frame(self.root, padding=24)
        frame.grid(row=0, column=0)

        fields = [
            ("age", "Age", "entry", None),
            ("bmi", "BMI", "entry", None),
            ("has_given_birth", "Has given birth", "combo", ("Unknown", "Yes", "No")),
            ("age_at_first_birth", "Age at first birth", "entry", None),
            ("is_postmenopausal", "Postmenopausal", "combo", ("Unknown", "Yes", "No")),
            ("menopause_type", "Menopause type", "combo", ("Unknown", "Natural", "Surgical")),
            ("currently_on_hrt", "Currently on HRT", "combo", ("Unknown", "Yes", "No")),
            (
                "first_degree_relatives_with_breast_cancer",
                "First-degree relatives with breast cancer",
                "combo",
                ("Unknown", "0", "1", "2+"),
            ),
            (
                "breast_density",
                "Breast density",
                "combo",
                (
                    "Unknown",
                    "Almost entirely fat",
                    "Scattered fibroglandular",
                    "Heterogeneously dense",
                    "Extremely dense",
                ),
            ),
            ("prior_breast_procedure", "Prior breast procedure", "combo", ("Unknown", "Yes", "No")),
            ("last_mammogram_result", "Last mammogram result", "combo", ("Unknown", "Negative", "False positive")),
        ]

        for row, (key, label, kind, options) in enumerate(fields):
            self.values[key] = tk.StringVar()
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 18), pady=4)
            if kind == "entry":
                widget: ttk.Entry | ttk.Combobox = ttk.Entry(frame, textvariable=self.values[key], width=30)
            else:
                widget = ttk.Combobox(
                    frame,
                    textvariable=self.values[key],
                    values=options,
                    state="readonly",
                    width=28,
                )
                widget.current(0)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        ttk.Button(frame, text="Submit", command=self._submit).grid(
            row=len(fields),
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(16, 8),
        )

        self.result = ttk.Label(frame, text="", style="Result.TLabel", anchor="center", justify="center")
        self.result.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(8, 2))
        self.error = ttk.Label(frame, text="", style="Error.TLabel", anchor="center", wraplength=360)
        self.error.grid(row=len(fields) + 2, column=0, columnspan=2, pady=(2, 0))

    def _optional_number(self, key: str) -> float | None:
        value = self.values[key].get().strip()
        return None if not value else float(value)

    def _choice(self, key: str, unknown: object = None) -> object:
        value = self.values[key].get()
        return unknown if value == "Unknown" else value

    def _patient(self) -> dict[str, object]:
        age_text = self.values["age"].get().strip()
        if not age_text:
            raise ValueError("Age is required.")

        has_given_birth = self._choice("has_given_birth")
        if has_given_birth == "Yes":
            has_given_birth = True
        elif has_given_birth == "No":
            has_given_birth = False

        is_postmenopausal = self._choice("is_postmenopausal")
        if is_postmenopausal == "Yes":
            is_postmenopausal = True
        elif is_postmenopausal == "No":
            is_postmenopausal = False

        menopause_type = self._choice("menopause_type")
        if menopause_type == "Natural":
            menopause_type = "natural"
        elif menopause_type == "Surgical":
            menopause_type = "surgical"

        currently_on_hrt = self._choice("currently_on_hrt")
        if currently_on_hrt == "Yes":
            currently_on_hrt = True
        elif currently_on_hrt == "No":
            currently_on_hrt = False

        density = self._choice("breast_density")
        if isinstance(density, str):
            density = density.lower()

        mammogram = self._choice("last_mammogram_result")
        if mammogram == "Negative":
            mammogram = "negative"
        elif mammogram == "False positive":
            mammogram = "false positive"

        prior = self._choice("prior_breast_procedure")
        if prior == "Yes":
            prior = True
        elif prior == "No":
            prior = False

        return {
            "age": int(age_text),
            "bmi": self._optional_number("bmi"),
            "has_given_birth": has_given_birth,
            "age_at_first_birth": self._optional_number("age_at_first_birth"),
            "is_postmenopausal": is_postmenopausal,
            "menopause_type": menopause_type,
            "currently_on_hrt": currently_on_hrt,
            "first_degree_relatives_with_breast_cancer": self._choice(
                "first_degree_relatives_with_breast_cancer"
            ),
            "breast_density": density,
            "prior_breast_procedure": prior,
            "last_mammogram_result": mammogram,
        }

    def _submit(self) -> None:
        self.result.configure(text="")
        self.error.configure(text="")
        try:
            output = predict_risk_score(self._patient())
        except (InputMappingError, SUPMTError, ValueError, TypeError, OSError) as exc:
            self.error.configure(text=str(exc))
            return

        probability = min(100.0, max(0.0, float(output["weighted_average_pct"])))
        score = min(100.0, max(0.0, float(output["risk_score"])))
        unknown_count = int(output["unknown_field_count"])
        self.result.configure(
            text=(
                f"Estimated Risk: {probability:.2f}%\n"
                f"Relative Risk Score: {score:.1f}/100\n"
                f"Unknown Fields: {unknown_count}"
            )
        )


def main() -> None:
    root = tk.Tk()
    RiskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
