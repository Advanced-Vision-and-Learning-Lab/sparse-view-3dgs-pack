# bench_tables.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import math
import csv
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BenchTables:
    """Create LaTeX tables + PNG table images from benchmark results."""
    # expected row schema (same you append to summary_rows)
    # {'tag','scene','model','seed','exit_code','duration_sec','psnr','ssim','lpips'}
    rows: list[dict] = field(default_factory=list)

    # ---------- constructors ----------
    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "BenchTables":
        csv_path = Path(csv_path)
        rows = []
        if csv_path.exists():
            with open(csv_path, "r") as f:
                for r in csv.DictReader(f):
                    # coerce numeric if present
                    for k in ("psnr", "ssim", "lpips", "duration_sec"):
                        if k in r and r[k] not in (None, "", "None"):
                            r[k] = float(r[k])
                        else:
                            r[k] = None
                    rows.append(r)
        return cls(rows)

    def extend(self, more_rows: list[dict]) -> None:
        self.rows.extend(more_rows)

    def export(self, out_root: str | Path, overall_title: str = "All Scenes (mean)") -> Path:
        """
        Writes:
        <out_root>/tables/overall.{tex,png}
        <out_root>/tables/<scene>.{tex,png}
        <out_root>/tables/<model>_categories.{tex,png}
        <out_root>/tables/combined.tex         (all tables in one file)
        """
        out_root = Path(out_root)
        out_dir = out_root / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Aggregate results by scene and overall ---
        by_scene, overall = self.aggregate(self.rows)

        # --- OVERALL TABLE ---
        overall_rows = self._rows_from_bucket(overall)
        overall_tex  = self._latex_table(overall_rows, "Comparison over all scenes (mean).", "tab:overall")
        (out_dir / "overall.tex").write_text(overall_tex)
        self._png_table(overall_rows, overall_title, out_dir / "overall.png")

        # --- Combine all LaTeX parts ---
        combined_parts = [self._latex_preamble()]
        combined_parts.append("% ---- OVERALL ----")
        combined_parts.append(overall_tex + "\n")

        # --- PER-SCENE TABLES ---
        for scene, bucket in by_scene.items():
            scene_rows = self._rows_from_bucket(bucket)
            caption = f"Comparison on {scene}."
            label   = f"tab:{scene}"
            tex_str = self._latex_table(scene_rows, caption, label)
            (out_dir / f"{scene}.tex").write_text(tex_str)
            self._png_table(scene_rows, scene, out_dir / f"{scene}.png")
            combined_parts.append(f"% ---- {scene} ----")
            combined_parts.append(tex_str + "\n")

        # --- PER-MODEL TABLES (your requested new block) ---
        combined_parts.append("% ---- PER-MODEL TABLES ----")

        # invert aggregation: model -> {scene -> metrics}
        models_seen = sorted({m for _, bucket in by_scene.items() for m in bucket})
        for model in models_seen:
            model_bucket = {}
            for scene, bucket in by_scene.items():
                if model in bucket:
                    # compute mean ± std for that model-scene pair
                    vals = bucket[model]
                    def _mean_std(vals, key):
                        import numpy as np
                        xs = [v[key] for v in vals if v.get(key) is not None]
                        if not xs: return None, None
                        arr = np.array(xs)
                        return float(arr.mean()), float(arr.std())
                    m_psnr, s_psnr = _mean_std(vals, "psnr")
                    m_ssim, s_ssim = _mean_std(vals, "ssim")
                    m_lpips, s_lpips = _mean_std(vals, "lpips")
                    model_bucket[scene] = {
                        "PSNR": (m_psnr, s_psnr),
                        "SSIM": (m_ssim, s_ssim),
                        "LPIPS": (m_lpips, s_lpips),
                    }

            # format rows (Category = Scene)
            scene_rows = [
                {"Scene": sc,
                "PSNR": v["PSNR"],
                "SSIM": v["SSIM"],
                "LPIPS": v["LPIPS"]}
                for sc, v in model_bucket.items()
            ]
            scene_rows.sort(key=lambda x: x["Scene"].lower())

            caption = f"Per-category reconstruction metrics for {model}."
            label   = f"tab:{model}_category_metrics"
            tex_str = self._latex_table_model(scene_rows, caption, label)
            (out_dir / f"{model}_categories.tex").write_text(tex_str)
            self._png_table_model(scene_rows, f"{model} (per-scene)", out_dir / f"{model}_categories.png")
            combined_parts.append(f"% ---- {model} ----")
            combined_parts.append(tex_str + "\n")

        # --- Final combined file ---
        combined_parts.append(self._latex_postamble())
        (out_dir / "combined.tex").write_text("\n".join(combined_parts))

        return out_dir

    # ---------- aggregation ----------
    @staticmethod
    def aggregate(rows):
        """
        Build:
        by_scene: {scene -> {model: [ {psnr, ssim, lpips}, ... ]}}
        overall : {model -> [ {psnr, ssim, lpips}, ... ]}
        """
        from collections import defaultdict
        by_scene = defaultdict(lambda: defaultdict(list))
        overall  = defaultdict(list)

        all_models = set()
        all_scenes = set()

        for r in rows:
            m = r.get("model"); s = r.get("scene")
            if not m or not s:
                continue
            all_models.add(m); all_scenes.add(s)
            rec = {k: r.get(k) for k in ("psnr", "ssim", "lpips")}
            if not all(v is None for v in rec.values()):
                by_scene[s][m].append(rec)
                overall[m].append(rec)

        # ensure placeholders so models show up even with no metrics
        for s in all_scenes:
            for m in all_models:
                by_scene[s].setdefault(m, by_scene[s].get(m, []))
        for m in all_models:
            overall.setdefault(m, overall.get(m, []))

        return by_scene, overall


    def _rows_from_bucket(self, bucket):
        """
        bucket: {model: [ {psnr, ssim, lpips}, ... ] }
        Shows mean only if one entry, mean ± std if multiple entries.
        """
        import numpy as np

        rows = []
        for model, arr in bucket.items():
            def _mean_or_pm(key):
                xs = [v[key] for v in arr if v.get(key) is not None]
                if not xs:
                    return None
                if len(xs) == 1:
                    return float(xs[0])  # single value — no std
                arr_np = np.array(xs)
                return (float(arr_np.mean()), float(arr_np.std()))  # multiple values — mean ± std

            rows.append({
                "Method": model,
                "PSNR": _mean_or_pm("psnr"),
                "SSIM": _mean_or_pm("ssim"),
                "LPIPS": _mean_or_pm("lpips"),
            })

        rows.sort(key=lambda x: x["Method"].lower())
        return rows


    @staticmethod
    def _mean(vals: list[dict], key: str):
        xs = [v[key] for v in vals if v.get(key) is not None]
        return sum(xs)/len(xs) if xs else None

    # def _rows_from_bucket(self, bucket: dict[str, list[dict]]):
    #     rows = []
    #     for model, arr in bucket.items():
    #         rows.append({
    #             "Method": model,
    #             "PSNR": self._mean(arr, "psnr"),
    #             "SSIM": self._mean(arr, "ssim"),
    #             "LPIPS": self._mean(arr, "lpips"),
    #         })
    #     rows.sort(key=lambda x: x["Method"].lower())
    #     return rows

    # ---------- LaTeX ----------

    @staticmethod
    def _fmt(x, kind):
        """
        Format scalar or (mean, std) tuple.
        Example: 18.12 ± 0.23
        """
        import math
        if x is None:
            return "--"
        if isinstance(x, tuple):
            mean, std = x
            if mean is None:
                return "--"
            if std is None or math.isnan(std):
                return f"{mean:.2f}" if kind == "psnr" else f"{mean:.3f}"
            # choose decimals per metric
            if kind == "psnr":
                return f"{mean:.2f} ± {std:.2f}"
            else:
                return f"{mean:.3f} ± {std:.3f}"
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return "--"
            return f"{x:.2f}" if kind == "psnr" else f"{x:.3f}"
        return str(x)


    @staticmethod
    def _best_indices(rows):
        idx_psnr = max(((i, r["PSNR"]) for i, r in enumerate(rows) if r["PSNR"] is not None), default=(None, None))[0]
        idx_ssim = max(((i, r["SSIM"]) for i, r in enumerate(rows) if r["SSIM"] is not None), default=(None, None))[0]
        idx_lpips = min(((i, r["LPIPS"]) for i, r in enumerate(rows) if r["LPIPS"] is not None), default=(None, None))[0]
        return idx_psnr, idx_ssim, idx_lpips

    def _latex_table(self, rows, caption, label):
        if not rows:
            return ""
        idx_psnr, idx_ssim, idx_lpips = self._best_indices(rows)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\setlength{\tabcolsep}{8pt}",
            r"\renewcommand{\arraystretch}{1.1}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
            r"\midrule",
        ]
        for i, r in enumerate(rows):
            ps = self._fmt(r.get("PSNR"), "psnr")
            ss = self._fmt(r.get("SSIM"), "ssim")
            lp = self._fmt(r.get("LPIPS"), "lpips")
            if i == idx_psnr and ps != "--": ps = rf"\textbf{{{ps}}}"
            if i == idx_ssim and ss != "--": ss = rf"\textbf{{{ss}}}"
            if i == idx_lpips and lp != "--": lp = rf"\textbf{{{lp}}}"
            lines.append(f"{r['Method']} & {ps} & {ss} & {lp} \\\\")
        lines += [
            r"\bottomrule",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)
    def _latex_table_model(self, rows, caption, label):
        """
        Per-model LaTeX table (rows = scenes).
        Scene rows: show mean only (no ±).
        Final row:  'Average' with mean ± std.
        """
        if not rows:
            return ""

        import numpy as np

        # helper: pull mean from scalar or (mean, std)
        def _mean_only(x, kind):
            if isinstance(x, tuple):
                x = x[0]
            if x is None:
                return "--"
            return f"{x:.2f}" if kind == "psnr" else f"{x:.3f}"

        # compute average row (mean ± std)
        def _extract_mean(x):
            if isinstance(x, tuple):
                return x[0]
            return x if isinstance(x, (int, float)) else None

        avg = {}
        for k in ("PSNR", "SSIM", "LPIPS"):
            vals = [_extract_mean(r.get(k)) for r in rows if _extract_mean(r.get(k)) is not None]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                avg[k] = (m, s)
            else:
                avg[k] = (None, None)

        # format mean ± std for average row
        def _pm(x, kind):
            if not x or x[0] is None:
                return "--"
            m, s = x
            if s is None or s < 1e-6:
                return f"{m:.2f}" if kind == "psnr" else f"{m:.3f}"
            return f"{m:.2f} ± {s:.2f}" if kind == "psnr" else f"{m:.3f} ± {s:.3f}"

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\setlength{\tabcolsep}{8pt}",
            r"\renewcommand{\arraystretch}{1.1}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Scene & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
            r"\midrule",
        ]

        # scene rows: mean only
        for r in rows:
            ps = _mean_only(r.get("PSNR"), "psnr")
            ss = _mean_only(r.get("SSIM"), "ssim")
            lp = _mean_only(r.get("LPIPS"), "lpips")
            lines.append(f"{r['Scene']} & {ps} & {ss} & {lp} \\\\")

        # average row: mean ± std and bold
        lines += [
            r"\midrule",
            rf"\textbf{{Average}} & \textbf{{{_pm(avg['PSNR'],'psnr')}}} & "
            rf"\textbf{{{_pm(avg['SSIM'],'ssim')}}} & "
            rf"\textbf{{{_pm(avg['LPIPS'],'lpips')}}} \\",
            r"\bottomrule",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _latex_preamble():
        return (
            r"% Combined tables" "\n"
            r"\providecommand{\tabcolsepdefault}{\tabcolsep}" "\n"
            r"\setlength{\tabcolsep}{8pt}" "\n"
        )

    @staticmethod
    def _latex_postamble():
        return r"\setlength{\tabcolsep}{\tabcolsepdefault}"
    

    def _png_table_model(self, rows, title, out_png: Path):
        """
        Create a PNG table for a single model across multiple scenes.
        Each row = scene/category, columns = PSNR, SSIM, LPIPS.
        Displays mean for per-scene rows, mean ± std only for final 'Average' row.
        Hides ± 0.00 in scene rows.
        """

        if not rows:
            return

        # --- Compute average row (mean ± std) ---
        def _extract_mean(x):
            if isinstance(x, tuple):
                return x[0]
            return x if isinstance(x, (int, float)) else None

        avg_row = {"Scene": "Average"}
        for k in ("PSNR", "SSIM", "LPIPS"):
            vals = [_extract_mean(r.get(k)) for r in rows if _extract_mean(r.get(k)) is not None]
            if vals:
                avg_row[k] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                avg_row[k] = (None, None)

        headers = ["Scene", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]
        data = []

        # --- Per-scene rows: only mean (no ± std, hide ± 0.00) ---
        def _val_str(x, kind):
            if isinstance(x, tuple):
                x = x[0]
            if x is None:
                return "--"
            return f"{x:.2f}" if kind == "psnr" else f"{x:.3f}"

        for r in rows:
            data.append([
                r["Scene"],
                _val_str(r.get("PSNR"), "psnr"),
                _val_str(r.get("SSIM"), "ssim"),
                _val_str(r.get("LPIPS"), "lpips"),
            ])

        # --- Average row: mean ± std ---
        def _fmt_pm(x, kind):
            if not x or x[0] is None:
                return "--"
            mean, std = x
            # Hide ± if std ~ 0
            if std < 1e-6:
                return f"{mean:.2f}" if kind == "psnr" else f"{mean:.3f}"
            if kind == "psnr":
                return f"{mean:.2f} ± {std:.2f}"
            else:
                return f"{mean:.3f} ± {std:.3f}"

        data.append([
            "Average",
            _fmt_pm(avg_row["PSNR"], "psnr"),
            _fmt_pm(avg_row["SSIM"], "ssim"),
            _fmt_pm(avg_row["LPIPS"], "lpips"),
        ])

        # --- Plot table ---
        n = max(1, len(data))
        fig_h = 0.7 + 0.38 * n
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        ax.axis("off")

        table = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.25)

        # --- Styling ---
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
            if row == len(data):
                cell.set_text_props(weight="bold")
            cell.set_linewidth(0.6)

        ax.set_title(title, fontsize=12, pad=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)


    def _png_table(self, rows, title, out_png: Path):
        n = max(1, len(rows))
        fig_h = 0.7 + 0.38 * n
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        ax.axis("off")

        headers = ["Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]
        data = [[
            r["Method"],
            self._fmt(r.get("PSNR"), "psnr"),
            self._fmt(r.get("SSIM"), "ssim"),
            self._fmt(r.get("LPIPS"), "lpips"),
        ] for r in rows]

        table = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.25)

        # Header styling: bold
        for (row, col), cell in table.get_celld().items():
            # row == 0 is header row in matplotlib's table when colLabels is set
            if row == 0:
                cell.set_text_props(weight="bold")
            cell.set_linewidth(0.6)

        ax.set_title(title, fontsize=12, pad=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
