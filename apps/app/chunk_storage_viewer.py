from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .chunk_layout import (
    ChunkLayout,
    build_chunk_layout,
    write_chunk_layout_cache,
)
from .chunk_visibility import build_chunk_visibility_index, write_chunk_visibility_cache
from .paths import ProjectPaths
from .sphere_map_store import SphereMapError, SphereMapSession, SphereMapStore
from .topology import DualTopology, generate_dual_topology, write_topology_cache


class ChunkStorageViewer:
    def __init__(self, parent: tk.Misc, paths: ProjectPaths) -> None:
        self.paths = paths
        self.window = tk.Toplevel(parent)
        self.window.title("球面分块、层级可见性与 Pack 检查器 v1.3.1")
        self.window.geometry("920x720")
        self.window.minsize(740, 560)

        self.frequency = tk.StringVar(value="16")
        self.map_name = tk.StringVar(value="planet_chunk_test_f16")
        self.status = tk.StringVar(value="先生成拓扑和稳定分块")
        self.summary = tk.StringVar(value="尚未生成")
        self.topology: DualTopology | None = None
        self.layout: ChunkLayout | None = None
        self.session: SphereMapSession | None = None
        self.store = SphereMapStore(paths.map_root)
        self.busy = False
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(root, text="生成与存储操作", padding=10)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="测试细分频率").grid(row=0, column=0, sticky="w")
        self.frequency_combo = ttk.Combobox(
            controls,
            textvariable=self.frequency,
            values=("1", "2", "4", "8", "16", "32", "64", "128"),
            state="readonly",
            width=24,
        )
        self.frequency_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.frequency_combo.bind("<<ComboboxSelected>>", self._frequency_changed)

        self.generate_button = ttk.Button(
            controls,
            text="生成拓扑、分块与层级索引",
            command=self.generate_layout,
        )
        self.generate_button.grid(row=2, column=0, sticky="ew")

        ttk.Separator(controls).grid(row=3, column=0, sticky="ew", pady=12)
        ttk.Label(controls, text="球面测试地图名称").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.map_name, width=26).grid(
            row=5, column=0, sticky="ew", pady=(2, 8)
        )
        self.create_button = ttk.Button(
            controls,
            text="创建或打开分块地图",
            command=self.create_or_open_map,
            state=tk.DISABLED,
        )
        self.create_button.grid(row=6, column=0, sticky="ew")

        self.write_button = ttk.Button(
            controls,
            text="写入跨区块测试状态",
            command=self.write_test_states,
            state=tk.DISABLED,
        )
        self.write_button.grid(row=7, column=0, sticky="ew", pady=(6, 0))

        self.verify_button = ttk.Button(
            controls,
            text="随机读取并验证全部区块",
            command=self.verify_map,
            state=tk.DISABLED,
        )
        self.verify_button.grid(row=8, column=0, sticky="ew", pady=(6, 0))

        self.compact_button = ttk.Button(
            controls,
            text="分析并整理 Pack 历史数据",
            command=self.compact_map,
            state=tk.DISABLED,
        )
        self.compact_button.grid(row=9, column=0, sticky="ew", pady=(6, 0))

        self.recover_button = ttk.Button(
            controls,
            text="检查/恢复中断的 Pack 整理",
            command=self.recover_compaction,
            state=tk.DISABLED,
        )
        self.recover_button.grid(row=10, column=0, sticky="ew", pady=(6, 0))

        ttk.Separator(controls).grid(row=11, column=0, sticky="ew", pady=12)
        ttk.Label(
            controls,
            text=(
                "本检查器验证的是实际文件链路：\n"
                "chunks.idx → index.bin → pack_xxxx.bin。\n\n"
                "Pack 整理只保留 index.bin 当前引用的区块版本，"
                "并提供中断恢复标记与旧文件回滚。"
            ),
            wraplength=230,
            justify=tk.LEFT,
        ).grid(row=12, column=0, sticky="w")

        result_frame = ttk.LabelFrame(root, text="分块与存储结果", padding=10)
        result_frame.grid(row=0, column=1, sticky="nsew")
        result_frame.rowconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)

        ttk.Label(result_frame, textvariable=self.summary, justify=tk.LEFT).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )
        self.output = tk.Text(result_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.output.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scroll.set)

        status = ttk.Label(
            self.window,
            textvariable=self.status,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=(8, 4),
        )
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _frequency_changed(self, _event: tk.Event) -> None:
        self.map_name.set(f"planet_chunk_test_f{self.frequency.get()}")
        self.topology = None
        self.layout = None
        self.session = None
        self.create_button.configure(state=tk.DISABLED)
        self.write_button.configure(state=tk.DISABLED)
        self.verify_button.configure(state=tk.DISABLED)
        self.compact_button.configure(state=tk.DISABLED)
        self.recover_button.configure(state=tk.DISABLED)
        self.summary.set("频率已改变，需要重新生成分块")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.generate_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        if busy:
            self.create_button.configure(state=tk.DISABLED)
            self.write_button.configure(state=tk.DISABLED)
            self.verify_button.configure(state=tk.DISABLED)
            self.compact_button.configure(state=tk.DISABLED)
            self.recover_button.configure(state=tk.DISABLED)
        elif self.layout is not None:
            self.create_button.configure(state=tk.NORMAL)
            self.recover_button.configure(state=tk.NORMAL)
            if self.session is not None:
                self.write_button.configure(state=tk.NORMAL)
                self.verify_button.configure(state=tk.NORMAL)
                self.compact_button.configure(state=tk.NORMAL)

    def generate_layout(self) -> None:
        if self.busy:
            return
        try:
            frequency = int(self.frequency.get())
        except ValueError:
            messagebox.showerror("错误", "细分频率无效", parent=self.window)
            return
        self._set_busy(True)
        self.status.set(f"正在生成 frequency={frequency} 的拓扑、分块与层级索引……")
        worker = threading.Thread(target=self._generate_worker, args=(frequency,), daemon=True)
        worker.start()
        self.window.after(50, self._poll_result)

    def _generate_worker(self, frequency: int) -> None:
        try:
            topology = generate_dual_topology(frequency, max_cells=10 * 128 * 128 + 2)
            topology_cache = write_topology_cache(topology, self.paths.map_root)
            layout = build_chunk_layout(topology, target_cells=256)
            layout_cache = write_chunk_layout_cache(layout, self.paths.map_root / ".topology")
            visibility = build_chunk_visibility_index(topology, layout)
            visibility_cache = write_chunk_visibility_cache(visibility, self.paths.map_root / ".topology")
            self.results.put((
                "layout",
                (
                    topology,
                    layout,
                    visibility,
                    topology_cache.directory,
                    layout_cache.index_path,
                    visibility_cache.index_path,
                ),
            ))
        except Exception as exc:
            self.results.put(("error", exc))

    def _poll_result(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value = self.results.get_nowait()
        except queue.Empty:
            if self.busy:
                self.window.after(50, self._poll_result)
            return
        self._set_busy(False)
        if state == "error":
            self.status.set("操作失败")
            messagebox.showerror("失败", str(value), parent=self.window)
            return
        if state == "layout":
            topology, layout, visibility, topology_dir, layout_path, visibility_path = value
            self.topology = topology
            self.layout = layout
            self.session = None
            validation = layout.validate(topology)
            sizes = [len(chunk.cell_ids) for chunk in layout.chunks]
            average = sum(sizes) / len(sizes)
            self.summary.set(
                f"Cell：{layout.cell_count:,}    区块：{layout.chunk_count:,}    "
                f"最小/平均/最大：{min(sizes)}/{average:.1f}/{max(sizes)}"
            )
            self._replace_output(
                "\n".join(
                    (
                        "稳定分块验证通过。",
                        f"细分频率：{layout.frequency}",
                        f"基础面：20",
                        f"目标区块容量：{layout.target_cells}",
                        f"连通区块：{'通过' if validation.connected_chunks else '失败'}",
                        f"拓扑哈希：{layout.topology_hash}",
                        f"分块哈希：{layout.stable_hash}",
                        f"层级节点：{visibility.node_count}",
                        f"层级索引哈希：{visibility.stable_hash}",
                        f"拓扑缓存：{topology_dir}",
                        f"分块索引：{layout_path}",
                        f"可见性索引：{visibility_path}",
                    )
                )
            )
            self.status.set("拓扑、分块和层级可见性缓存已写入")
            self.create_button.configure(state=tk.NORMAL)
            self.recover_button.configure(state=tk.NORMAL)

    def create_or_open_map(self) -> None:
        layout = self.layout
        if layout is None:
            return
        name = self.map_name.get().strip()
        try:
            recovery = self.store.compaction_recovery_state(name)
            if recovery.required:
                self.recover_button.configure(state=tk.NORMAL)
                messagebox.showwarning(
                    "需要恢复 Pack 整理",
                    f"地图存在未完成的 Pack 整理事务。\n阶段：{recovery.phase}\n"
                    "请先点击“检查/恢复中断的 Pack 整理”。",
                    parent=self.window,
                )
                return
            if name in self.store.list_maps():
                session = self.store.open(name, layout)
                action = "已打开"
            else:
                session = self.store.create_blank(name, layout)
                action = "已创建"
        except (SphereMapError, OSError) as exc:
            messagebox.showerror("地图存储失败", str(exc), parent=self.window)
            return
        self.session = session
        self.write_button.configure(state=tk.NORMAL)
        self.verify_button.configure(state=tk.NORMAL)
        self.compact_button.configure(state=tk.NORMAL)
        self.recover_button.configure(state=tk.NORMAL)
        stats = self._storage_stats(session.map_dir)
        analysis = self.store.analyze_storage(session)
        self._append_output(
            "\n\n"
            + "\n".join(
                (
                    f"{action}球面分块地图：{session.name}",
                    f"index.bin：{stats['index_bytes']:,} 字节",
                    f"Pack 文件：{stats['pack_count']} 个",
                    f"Pack 总量：{stats['pack_bytes']:,} 字节",
                    f"当前有效数据：{analysis.live_bytes:,} 字节",
                    f"可回收历史数据：{analysis.reclaimable_bytes:,} 字节",
                    f"历史失效区块块体：{analysis.orphan_blocks}",
                    f"地图目录：{session.map_dir}",
                )
            )
        )
        self.status.set(f"{action}地图 {session.name}")

    def write_test_states(self) -> None:
        session = self.session
        if session is None:
            return
        editable_cells = [cell_id for cell_id in range(12, session.layout.cell_count)]
        if not editable_cells:
            messagebox.showinfo(
                "没有可编辑六边形",
                "frequency=1 只有 12 个保留五边形，请选择 frequency=2 或更高值。",
                parent=self.window,
            )
            return
        candidates = [
            editable_cells[0],
            editable_cells[len(editable_cells) // 2],
            editable_cells[-1],
        ]
        try:
            for rotation, cell_id in enumerate(candidates):
                session.set_cell(
                    cell_id,
                    self.store,
                    "00000000-0000-0000-0000-000000000003",
                    "测试/阶段三测试笔刷.png",
                    rotation * 2,
                )
            dirty_before = tuple(sorted(session.dirty_chunks))
            old_sizes = self._pack_sizes(session.map_dir)
            self.store.save(session)
            new_sizes = self._pack_sizes(session.map_dir)
            reopened = self.store.open(session.name, session.layout)
            results = [reopened.brush_uid_for_cell(cell_id, self.store) for cell_id in candidates]
            self.session = reopened
        except (SphereMapError, OSError) as exc:
            messagebox.showerror("写入失败", str(exc), parent=self.window)
            return

        changed_packs = [
            pack_id
            for pack_id, size in new_sizes.items()
            if size != old_sizes.get(pack_id, 0)
        ]
        self._append_output(
            "\n\n"
            + "\n".join(
                (
                    f"增量保存区块：{', '.join(map(str, dirty_before))}",
                    f"实际追加 Pack：{', '.join(map(str, changed_packs))}",
                    f"重新打开读取：{results}",
                    "旧区块块体保留，新 index.bin 已原子替换。",
                )
            )
        )
        self.status.set("跨区块状态已增量保存并重新读取")

    def verify_map(self) -> None:
        session = self.session
        if session is None or self.busy:
            return
        self._set_busy(True)
        self.status.set("正在随机读取并校验全部区块……")
        worker = threading.Thread(target=self._verify_worker, args=(session,), daemon=True)
        worker.start()
        self.window.after(50, self._poll_verify)

    def _verify_worker(self, session: SphereMapSession) -> None:
        try:
            report = self.store.verify(session)
            self.results.put(("verify", report))
        except Exception as exc:
            self.results.put(("error", exc))

    def _poll_verify(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value = self.results.get_nowait()
        except queue.Empty:
            if self.busy:
                self.window.after(50, self._poll_verify)
            return
        self._set_busy(False)
        if state == "error":
            self.status.set("验证失败")
            messagebox.showerror("验证失败", str(value), parent=self.window)
            return
        report = value
        self._append_output(
            "\n\n"
            + "\n".join(
                (
                    f"全部区块读取验证：{'通过' if report.valid else '失败'}",
                    f"检查区块：{report.checked_chunks}",
                    f"损坏区块：{list(report.failed_chunks)}",
                    *(report.issues[:10]),
                )
            )
        )
        self.status.set("全部区块读取验证完成")

    def compact_map(self) -> None:
        session = self.session
        if session is None or self.busy:
            return
        if session.dirty_chunks or session.brush_table_dirty:
            messagebox.showwarning(
                "需要先保存",
                "Pack 整理前必须先保存全部脏区块和笔刷表修改。",
                parent=self.window,
            )
            return
        try:
            recovery = self.store.compaction_recovery_state(session.name)
            if recovery.required:
                messagebox.showwarning(
                    "需要恢复",
                    f"当前存在未完成的整理事务：{recovery.phase}",
                    parent=self.window,
                )
                return
            analysis = self.store.analyze_storage(session)
        except (SphereMapError, OSError) as exc:
            messagebox.showerror("分析失败", str(exc), parent=self.window)
            return
        if analysis.reclaimable_bytes <= 0 and analysis.orphan_blocks <= 0:
            messagebox.showinfo(
                "无需整理",
                "当前 Pack 没有可回收的历史区块数据。",
                parent=self.window,
            )
            return
        if not messagebox.askyesno(
            "整理 Pack",
            (
                f"当前 Pack：{analysis.physical_bytes:,} 字节\n"
                f"预计可回收：{analysis.reclaimable_bytes:,} 字节\n"
                f"历史失效块体：{analysis.orphan_blocks}\n\n"
                "程序会先生成并验证新 Pack，再切换正式文件。继续吗？"
            ),
            parent=self.window,
        ):
            return
        self._set_busy(True)
        self.status.set("正在重写、验证并原子切换 Pack……")
        threading.Thread(target=self._compact_worker, args=(session,), daemon=True).start()
        self.window.after(50, self._poll_compact)

    def _compact_worker(self, session: SphereMapSession) -> None:
        try:
            report = self.store.compact(session)
            self.results.put(("compact", report))
        except Exception as exc:
            self.results.put(("error", exc))

    def _poll_compact(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value = self.results.get_nowait()
        except queue.Empty:
            if self.busy:
                self.window.after(50, self._poll_compact)
            return
        self._set_busy(False)
        if state == "error":
            self.status.set("Pack 整理失败")
            messagebox.showerror("Pack 整理失败", str(value), parent=self.window)
            return
        report = value
        self._append_output(
            "\n\n"
            + "\n".join(
                (
                    "Pack 历史数据整理完成。",
                    f"整理前：{report.before.physical_bytes:,} 字节",
                    f"整理后：{report.after.physical_bytes:,} 字节",
                    f"实际回收：{report.bytes_reclaimed:,} 字节",
                    f"移除失效块体：{report.before.orphan_blocks - report.after.orphan_blocks}",
                    f"验证区块：{report.verified_chunks}",
                    "index.bin 与全部 Pack 已重新验证。",
                )
            )
        )
        self.status.set(f"Pack 整理完成，回收 {report.bytes_reclaimed:,} 字节")

    def recover_compaction(self) -> None:
        layout = self.layout
        if layout is None or self.busy:
            return
        name = self.map_name.get().strip()
        try:
            state = self.store.compaction_recovery_state(name)
        except SphereMapError as exc:
            messagebox.showerror("检查失败", str(exc), parent=self.window)
            return
        if not state.required:
            messagebox.showinfo("无需恢复", "没有检测到中断的 Pack 整理事务。", parent=self.window)
            return
        if state.phase == "orphaned_artifacts":
            messagebox.showerror(
                "无法自动恢复",
                state.details + "\n为避免误删文件，当前版本不会在缺少事务标记时自动处理。",
                parent=self.window,
            )
            return
        if not messagebox.askyesno(
            "恢复 Pack 整理",
            f"检测到中断阶段：{state.phase}。\n程序会验证新文件；不完整时自动回滚旧文件。继续吗？",
            parent=self.window,
        ):
            return
        self._set_busy(True)
        self.status.set("正在恢复中断的 Pack 整理……")
        threading.Thread(target=self._recover_worker, args=(name, layout), daemon=True).start()
        self.window.after(50, self._poll_recover)

    def _recover_worker(self, name: str, layout: ChunkLayout) -> None:
        try:
            report = self.store.recover_compaction(name, layout)
            session = self.store.open(name, layout)
            self.results.put(("recover", (report, session)))
        except Exception as exc:
            self.results.put(("error", exc))

    def _poll_recover(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value = self.results.get_nowait()
        except queue.Empty:
            if self.busy:
                self.window.after(50, self._poll_recover)
            return
        self._set_busy(False)
        if state == "error":
            self.status.set("Pack 恢复失败")
            messagebox.showerror("Pack 恢复失败", str(value), parent=self.window)
            return
        report, session = value
        self.session = session
        self.write_button.configure(state=tk.NORMAL)
        self.verify_button.configure(state=tk.NORMAL)
        self.compact_button.configure(state=tk.NORMAL)
        action_names = {
            "finalized_new": "保留并完成新 Pack",
            "rolled_back": "回滚到旧 Pack",
            "discarded_stage": "删除未完成临时 Pack 并保留原地图",
            "none": "无需处理",
        }
        action = action_names.get(report.action, report.action)
        self._append_output(
            "\n\n"
            + "\n".join(
                (
                    f"中断恢复结果：{action}",
                    f"验证区块：{report.verified_chunks}",
                    "恢复标记和临时文件已经清理。",
                )
            )
        )
        self.status.set(f"Pack 恢复完成：{action}")

    @staticmethod
    def _storage_stats(map_dir: Path) -> dict[str, int]:
        pack_files = sorted((map_dir / "data").glob("pack_*.bin"))
        return {
            "index_bytes": (map_dir / "index.bin").stat().st_size,
            "pack_count": len(pack_files),
            "pack_bytes": sum(path.stat().st_size for path in pack_files),
        }

    @staticmethod
    def _pack_sizes(map_dir: Path) -> dict[int, int]:
        result: dict[int, int] = {}
        for path in (map_dir / "data").glob("pack_*.bin"):
            try:
                pack_id = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            result[pack_id] = path.stat().st_size
        return result

    def _replace_output(self, value: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", value)
        self.output.configure(state=tk.DISABLED)

    def _append_output(self, value: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, value)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)
