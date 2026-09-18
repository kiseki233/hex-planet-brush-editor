from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .paths import ProjectPaths
from .production_layout import (
    ProductionChunkLayout,
    load_production_layout_cache,
    write_production_layout_cache,
)
from .production_topology import ProductionTopology
from .production_visibility import (
    build_production_visibility_index,
    load_production_visibility_cache,
    write_production_visibility_cache,
)
from .sphere_map_store import SphereMapStore


class ProductionBuilder:
    def __init__(self, parent: tk.Misc, paths: ProjectPaths) -> None:
        self.paths = paths
        self.store = SphereMapStore(paths.map_root)
        self.window = tk.Toplevel(parent)
        self.window.title("千万格生产拓扑与地图构建器 v1.3.1")
        self.window.geometry("760x610")
        self.window.minsize(660, 520)
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.layout: ProductionChunkLayout | None = None

        self.frequency = tk.IntVar(value=1004)
        self.tile_side = tk.IntVar(value=16)
        self.map_name = tk.StringVar(value="planet_production_f1004")
        self.status = tk.StringVar(value="尚未加载生产布局")
        self.summary = tk.StringVar(value="")

        self._build_ui()
        self.window.after(80, self._poll)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(7, weight=1)

        ttk.Label(root, text="细分频率").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(root, from_=1, to=4096, textvariable=self.frequency, width=14).grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        ttk.Label(root, text="面内分块边长").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(root, from_=2, to=255, textvariable=self.tile_side, width=14).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0)
        )
        ttk.Label(root, text="地图名称").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(root, textvariable=self.map_name).grid(
            row=2, column=1, sticky="ew", padx=(10, 0), pady=(8, 0)
        )

        button_frame = ttk.Frame(root)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        for column in range(3):
            button_frame.columnconfigure(column, weight=1)
        self.build_button = ttk.Button(
            button_frame, text="生成/加载生产索引", command=self.build_layout
        )
        self.build_button.grid(row=0, column=0, sticky="ew")
        self.create_button = ttk.Button(
            button_frame, text="创建完整空白 Pack 地图", command=self.create_map, state=tk.DISABLED
        )
        self.create_button.grid(row=0, column=1, sticky="ew", padx=6)
        self.verify_button = ttk.Button(
            button_frame, text="逐区块验证地图", command=self.verify_map, state=tk.DISABLED
        )
        self.verify_button.grid(row=0, column=2, sticky="ew")

        ttk.Separator(root).grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(root, textvariable=self.status).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(root, textvariable=self.summary, justify=tk.LEFT).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 8)
        )
        self.log = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
        self.log.grid(row=7, column=0, columnspan=2, sticky="nsew")

        ttk.Label(
            root,
            text=(
                "该构建器不会展开一千万条 Python 几何对象。CellId、中心、邻居和六边形角点按需计算；"
                "磁盘只保存紧凑区块描述、生产可见性层级和 Pack 地图。可在主程序中打开千万格生产 GPU 编辑器。"
            ),
            wraplength=710,
            justify=tk.LEFT,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.build_button.configure(state=state)
        self.create_button.configure(state=(tk.DISABLED if running or self.layout is None else tk.NORMAL))
        self.verify_button.configure(state=(tk.DISABLED if running or self.layout is None else tk.NORMAL))

    def _append(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def build_layout(self) -> None:
        if self.running:
            return
        try:
            frequency = int(self.frequency.get())
            tile_side = int(self.tile_side.get())
        except (TypeError, ValueError):
            messagebox.showerror("错误", "频率和分块边长必须是整数")
            return
        self._set_running(True)
        self.status.set(f"正在准备 frequency={frequency} 的生产索引……")
        threading.Thread(
            target=self._build_worker, args=(frequency, tile_side), daemon=True
        ).start()

    def _build_worker(self, frequency: int, tile_side: int) -> None:
        try:
            started = time.perf_counter()
            cache_dir = self.paths.map_root / ".topology" / f"ico_dual_f{frequency}_v2"
            if cache_dir.exists():
                layout = load_production_layout_cache(cache_dir)
                if layout.tile_side != tile_side:
                    raise ValueError(
                        f"现有缓存 tileSide={layout.tile_side}，与输入的 {tile_side} 不一致"
                    )
                source = "读取缓存"
            else:
                topology = ProductionTopology(frequency)
                layout = ProductionChunkLayout(topology, tile_side=tile_side)
                write_production_layout_cache(layout, self.paths.map_root / ".topology")
                source = "重新生成"
            elapsed = time.perf_counter() - started
            visibility_path = cache_dir / "production_visibility.json"
            if visibility_path.exists():
                visibility = load_production_visibility_cache(cache_dir, layout)
                visibility_source = "读取可见性缓存"
            else:
                visibility = build_production_visibility_index(layout)
                write_production_visibility_cache(visibility, self.paths.map_root / ".topology")
                visibility_source = "生成可见性缓存"
            validation = layout.topology.validate()
            layout_validation = layout.validate()
            visibility_validation = visibility.validate(layout)
            if not validation.valid or not layout_validation.valid or not visibility_validation.valid:
                issues = (*validation.issues, *layout_validation.issues, *visibility_validation.issues)
                raise ValueError("；".join(issues[:8]))
            self.results.put((
                "layout",
                (layout, visibility, source, visibility_source, elapsed, layout_validation),
            ))
        except Exception as exc:
            self.results.put(("error", exc))

    def create_map(self) -> None:
        if self.running or self.layout is None:
            return
        name = self.map_name.get().strip()
        if not name:
            messagebox.showerror("错误", "地图名称不能为空")
            return
        self._set_running(True)
        self.status.set("正在创建完整空白 Pack 地图……")
        threading.Thread(target=self._create_worker, args=(name, self.layout), daemon=True).start()

    def _create_worker(self, name: str, layout: ProductionChunkLayout) -> None:
        try:
            started = time.perf_counter()
            if name in self.store.list_maps():
                session = self.store.open(name, layout)
                action = "已存在并成功打开"
            else:
                session = self.store.create_blank(name, layout)
                action = "已创建"
            self.results.put(("map", (session, action, time.perf_counter() - started)))
        except Exception as exc:
            self.results.put(("error", exc))

    def verify_map(self) -> None:
        if self.running or self.layout is None:
            return
        name = self.map_name.get().strip()
        self._set_running(True)
        self.status.set("正在逐区块读取和验证地图……")
        threading.Thread(target=self._verify_worker, args=(name, self.layout), daemon=True).start()

    def _verify_worker(self, name: str, layout: ProductionChunkLayout) -> None:
        try:
            started = time.perf_counter()
            session = self.store.open(name, layout)
            report = self.store.verify(session)
            self.results.put(("verify", (report, time.perf_counter() - started)))
        except Exception as exc:
            self.results.put(("error", exc))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "error":
                    self._set_running(False)
                    self.status.set("操作失败")
                    self._append(f"错误：{payload}")
                    messagebox.showerror("错误", str(payload))
                elif kind == "layout":
                    layout, visibility, source, visibility_source, elapsed, validation = payload
                    self.layout = layout
                    sizes = [record.cell_count for record in layout.records]
                    self._set_running(False)
                    self.status.set("生产索引已就绪")
                    self.summary.set(
                        f"CellId：{layout.cell_count:,}\n"
                        f"普通六边形：{layout.topology.hexagon_count:,}\n"
                        f"逻辑区块：{layout.chunk_count:,}\n"
                        f"区块大小：{min(sizes)}～{max(sizes)}，平均 {sum(sizes)/len(sizes):.3f}\n"
                        f"展开的单格/三角记录：0 / 0"
                    )
                    self._append(
                        f"{source}完成，{visibility_source}：frequency={layout.frequency}，"
                        f"区块={layout.chunk_count:,}，可见性节点={visibility.node_count:,}，"
                        f"抽样检查={validation.checked_chunks}，耗时={elapsed:.3f}s"
                    )
                elif kind == "map":
                    session, action, elapsed = payload
                    self._set_running(False)
                    self.status.set(f"地图 {action}")
                    self._append(
                        f"地图 {session.name} {action}：index记录={len(session.index_records):,}，"
                        f"耗时={elapsed:.3f}s"
                    )
                elif kind == "verify":
                    report, elapsed = payload
                    self._set_running(False)
                    self.status.set("地图验证通过" if report.valid else "地图验证失败")
                    self._append(
                        f"逐区块验证：valid={report.valid}，checked={report.checked_chunks:,}，"
                        f"failed={len(report.failed_chunks)}，耗时={elapsed:.3f}s"
                    )
                    for issue in report.issues:
                        self._append(f"  {issue}")
        except queue.Empty:
            pass
        if self.window.winfo_exists():
            self.window.after(80, self._poll)
