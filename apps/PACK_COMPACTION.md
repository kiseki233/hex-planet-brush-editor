# Pack Compaction and Recovery v0.8.0

## Why compaction exists

Normal map saving is append-only at the Pack level. A dirty chunk is appended as a new block and `index.bin` is atomically replaced to point at the new block. This protects the previously valid block during saving, but repeated edits leave historical blocks that are no longer referenced.

v0.8.0 can measure and reclaim those unreferenced historical blocks.

## Authoritative data

Compaction keeps only the block currently referenced by `index.bin` for every `ChunkId`.

The following are not changed:

- `map.json`
- `brush_table.json`
- cell state values
- local brush IDs
- topology identity
- chunk layout identity
- Pack count and chunk-to-Pack assignment

## Compaction sequence

1. Refuse compaction while dirty chunks or brush-table changes are unsaved.
2. Read every currently indexed chunk through the normal Pack decoder.
3. Validate Pack metadata, zlib data, CRC32, cell count, reserved bit, rotation, and brush-table references.
4. Write a complete new Pack set under `.pack_compaction_stage/`.
5. Write a new `index.bin` for the staged Pack set.
6. Reopen and verify every staged chunk.
7. Write `.pack_compaction.json` as a recovery marker.
8. Move the old `data/` and `index.bin` to fixed backup paths.
9. Install the verified new `data/` and `index.bin`.
10. Verify every installed chunk again.
11. Remove backups and the recovery marker.

The original Pack set is not deleted before the new Pack set has been completely written and verified.

## Recovery artifacts

During the short switch transaction, these items may exist:

```text
<map>/
├─ .pack_compaction.json
├─ .pack_compaction_stage/
├─ .pack_compaction_backup_data/
└─ .pack_compaction_backup_index.bin
```

If the program is interrupted:

- Before the new index is installed, recovery rolls back to the old Pack set.
- After the new data and index are installed, recovery verifies the new pair. A valid pair is retained; an invalid pair is rolled back.
- Files are never automatically deleted when backup artifacts exist without a valid transaction marker. This avoids guessing about user files.

## User interface

Open `打开分块与 Pack 检查器` from the main window.

Available operations:

- `分析并整理 Pack 历史数据`
- `检查/恢复中断的 Pack 整理`

The checker reports physical bytes, currently referenced bytes, estimated reclaimable bytes, and historical block count.

## Command line

```powershell
pwsh.exe -File apps/tools/build_chunk_map.ps1 `
  --frequency 32 `
  --map-name planet_chunk_f32 `
  --analyze-pack
```

Compact and verify:

```powershell
pwsh.exe -File apps/tools/build_chunk_map.ps1 `
  --frequency 32 `
  --map-name planet_chunk_f32 `
  --compact-pack `
  --verify-map
```

Recover an interrupted transaction:

```powershell
pwsh.exe -File apps/tools/build_chunk_map.ps1 `
  --frequency 32 `
  --map-name planet_chunk_f32 `
  --recover-pack `
  --verify-map
```

## Scope boundary

This is Pack history reclamation and interruption recovery. It is not a general damaged-map reconstruction system. If the currently indexed authoritative block is corrupt and no valid transaction backup exists, compaction stops and reports the damaged chunk instead of inventing replacement data.
