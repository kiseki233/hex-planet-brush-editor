# Final Acceptance v1.3.1

GUI:

```text
Main window → 运行最终验收与 Windows 诊断
```

Command line:

```powershell
pwsh.exe -File apps/tools/run_acceptance.ps1
```

Checks:

1. Root contains only `apps` and `art`.
2. Launch scripts and Unicode resource roots exist.
3. Brush catalog contains no invalid source images.
4. Frequency-1004 production layout and visibility indexes validate.
5. A complete 39,060-chunk f1004 Pack map is created and every chunk reread.
6. The same production map transitions through LOD4, LOD3, LOD2, LOD1, and LOD0 without retaining Pack chunks in the session cache.
7. Dirty Pack save/reopen, CRC, history analysis, compaction, and post-compaction verification pass.
8. Four brush LODs and selected/full multi-level aggregate cache paths pass.
9. Shared topology edges, four-pixel UV padding, and six rotations pass numerical validation.
10. Random folder brush groups, six rotations, complete overwrite, continuous gap-filled strokes, and diameter 1–500 validate.
11. The manual-scale cropper exports multiple distinct 512×512 RGBA PNG tiles to `art/data/` without registering them as brushes or overwriting files.
12. The 20-face 2D net shares stable CellIds and the same Pack map with the spherical editor, including duplicated seam copies.
13. 256 distinct LOD0 layers account for 276,889,600 RGBA bytes; released slots must be reused.
14. Windows/WGL diagnostics pass on the target PC or are explicitly marked `needs_windows` elsewhere.

Reports:

```text
apps/runtime/logs/acceptance_report.json
apps/runtime/logs/acceptance_report.txt
apps/runtime/logs/seam_validation.json
apps/runtime/logs/texture_stress.json
apps/runtime/logs/windows_diagnostic.json
apps/runtime/logs/windows_diagnostic.txt
```

The overall report is valid when there are zero `fail` items. `needs_windows` remains visible and must not be represented as a pass.
