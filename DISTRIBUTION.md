# Distribution and runtime guide

This document is the tracked public contract for the DeepFaceLab modernization package: what the package generates and selects, what the host and user must supply, how commands are launched, and which compatibility work is inside or outside the current baseline.

## Support boundary

Current validation covers the current Windows development environment and the current generated runtimes. It includes all four runtime variants, runtime selection and package isolation, launching from a caller working directory other than the repository, current-environment CPU/CUDA integration, GUI/XSeg Editor integration, and external-resource policy composition.

This is not a claim that the package works on every Windows installation. The following release and portability scenarios are not yet qualified:

- Clean-machine/clean-VM qualification.
- A different Windows user profile and clean `PATH`.
- A physically moved installation, including a moved path containing spaces.
- Clean-environment proof that normal packaged application execution does not depend on host/developer Python or developer Qt.
- Proof that a system CUDA Toolkit is unnecessary where the runtime contract does not require it.
- NVIDIA driver prerequisite-matrix validation.
- Microsoft Visual C++ prerequisite-matrix validation.
- Full offline/reproducible construction from release-package inputs.
- Full packaged-runtime end-to-end execution: extract, train, merge, and export/result-video.
- Moved GUI-runtime and XSeg Editor validation.
- Release-archive assembly, integrity, and fresh-extraction validation.
- Final provenance, privacy, license, and redistribution review.
- Final release qualification before treating the packaged baseline as frozen.

Caller-working-directory independence is already supported, but it is not the same as moving the installation or proving release portability.

## Ownership and packaging

| Component | Owner/source | Package contract |
| --- | --- | --- |
| Application source and launchers | Distribution | Tracked in the repository |
| Generated Python runtime | Runtime builder | Versioned under `runtime\versions`; selected at activation time |
| Python packages, Torch, and GUI libraries | Generated runtime | Installed from source-controlled locks and builder inputs |
| `ffmpeg` and `ffprobe` | Host system | Required on `PATH`; not bundled |
| NVIDIA GPU and driver | Host system | Required only by CUDA variants |
| Workspace media, faces, and models | User | User-managed; never runtime content |
| Generic-XSeg model | User | Optional external resource; not bundled or downloaded |
| Pretraining faceset | User | External resource; not bundled or downloaded |

Generated runtime/distribution-owned runtime trees must not contain user aligned faces, source or destination media, model checkpoints, previews, training history, or other workspace data. Keep those items in user-managed locations.

## System prerequisites

The supported host baseline is 64-bit Windows with the Microsoft Visual C++ 2015–2022 x64 runtime. CUDA variants additionally need a supported NVIDIA environment, including a compatible driver and GPU. A specific GPU model is not a general requirement.

`ffmpeg` and `ffprobe` are external system prerequisites. They must resolve through the sanitized application `PATH`; the first matching executable in `PATH` is used. They are not copied into generated runtimes, and this contract does not prescribe a fixed installation directory or build.

Normal application use does not require or fall back to a system Python installation. Runtime construction is separate and requires compatible host build tooling, including CPython 3.11 or newer under the current builder contract. The setup launcher searches the supported host Python entrypoints or can use the configured setup interpreter.

Historical avecl/OpenCL setup is not part of the current production/runtime contract and is not a package prerequisite.

## Runtime variants

The builder produces one of four explicit variants:

| Variant | Torch contract | GUI contract | Intended use |
| --- | --- | --- | --- |
| `cpu-nogui` | CPU build; CUDA unavailable | No PyQt workflow support | Non-GUI CPU workflows |
| `cpu-gui` | CPU build; CUDA unavailable | Packaged PyQt/Qt | CPU workflows including supported GUIs |
| `cuda-nogui` | CUDA-enabled build | No PyQt workflow support | Non-GUI workloads on supported NVIDIA systems |
| `cuda-gui` | CUDA-enabled build | Packaged PyQt/Qt | NVIDIA workloads including supported GUIs |

Variant selection is explicit; runtime construction does not silently choose a variant through GPU detection. NOGUI variants are not suitable for XSeg Editor or other GUI workflows.

All four variants currently use `onnxruntime` 1.30.0 with `CPUExecutionProvider`. A CUDA Torch runtime does not imply `onnxruntime-gpu` or `CUDAExecutionProvider` support.

GUI variants use their packaged PyQt/Qt libraries and plugin tree. Users should not point them at an arbitrary system Qt plugin directory. For XSeg Editor, Torch-first ordering is required by the current compatibility implementation. The exact lower-level Windows DLL interaction was not fully isolated.

## Runtime construction and activation

Runtime construction is a developer/build concern. From the repository root, choose the needed variant explicitly:

```bat
launchers\dfl-setup-runtime.bat cpu-gui --activate
```

The setup flow supports online construction from pinned inputs and a strict offline mode using prepared offline inputs and the artifact cache. Its accepted form is:

```bat
launchers\dfl-setup-runtime.bat <VARIANT> [--activate] [--offline] ^
  [--offline-inputs "<OFFLINE_INPUTS_DIR>"] ^
  [--artifact-cache "<ARTIFACT_CACHE_DIR>"] ^
  [--lock-timeout "<SECONDS>"]
```

Generated runtimes live as retained versions under `runtime\versions`. Activation verifies the target and atomically updates `runtime\active-runtime.txt`. The selector contains the active runtime identity; users should use the setup/builder activation operations rather than hand-editing the selector or manifests.

Build, activation, and rollback do not overwrite a retained runtime version. The low-level builder's rollback operation can select a retained, verified prior version; it verifies both the requested target and the retained current version before changing the selector. Runtime identities are content-derived, so public instructions intentionally do not hard-code machine-specific IDs.

Runtime manifests and identities support variant/package consistency, installed-tree integrity checks, runtime-content identity, and privacy/cache validation. They are not signed release provenance, tamper-proof security, or a claim of cryptographic release authenticity.

Generated runtimes are immutable package outputs. Do not run `pip install` into them or manually alter their files. Make dependency changes in the source-controlled requirement locks or builder inputs, then rebuild canonically.

## Application launcher

Use `launchers\dfl.bat` as the canonical application boundary:

```bat
launchers\dfl.bat --help
```

The launcher derives the repository root from its own location, changes to that root, strictly resolves the active-runtime selector, and starts the selected packaged interpreter in isolated mode. It forwards application arguments without reinterpretation and returns the application's exit code. Commands therefore do not depend on the caller's current working directory.

There is no system-Python fallback. During ordinary application launch, `dfl.bat` validates the active selector and the selected runtime directory, packaged interpreter, and repository bootstrap required to start the application. Missing or malformed launch-boundary state fails instead of silently running under another Python environment. Normal launch does not hash or re-verify the complete runtime manifest and installed tree on every invocation. Supported commands should go through `dfl.bat`, not an arbitrary direct invocation of packaged Python.

Full manifest and installed-tree verification belongs to the canonical runtime lifecycle. Runtime construction verifies before promotion; explicit runtime verification uses the builder's canonical verifier; activation and rollback verify the affected retained runtimes before changing the selector; and `dfl-envreport.bat --verify` applies the canonical runtime checks together with its environment and prerequisite diagnostics. These integrity checks do not constitute signed release provenance or release-portability certification.

Focused diagnostics are available through:

```bat
launchers\dfl.bat --self-test
launchers\dfl-envreport.bat --verify
```

The self-test checks the packaged bootstrap path. The environment report verifies the current environment, external tools, and runtime consistency and returns failure when required checks do not pass. These diagnostics are not the full automated test suite and do not certify release portability.

## Curated workflow launchers

The `.bat` files in `launchers` provide convenient, stable entrypoints for frequently used workflows. They apply conventional workspace paths and route execution through the selected packaged runtime. Some commands accept forwarded user arguments; use each command's help when supplying paths and quote paths that may contain spaces.

The curated set does not provide a dedicated `.bat` file for every historical launcher. A historical behavior can be reproduced by the generic `dfl.bat` command surface without duplicating it as a fixed-path wrapper.

## Workspace ownership and privacy

The workspace is user-managed data, conventionally:

```text
workspace\data_src
workspace\data_dst
workspace\model
```

Launchers and application workflows reference those locations relative to the repository root. Invoked workflows may create their expected output or model directories. Runtime construction does not ingest or package the workspace.

Keep backups according to the value of your media and models. Runtime activation and rollback govern generated runtimes; they are not workspace backup or rollback operations.

## XSeg resources and workflows

### Manual XSeg Editor

Manual mask editing uses aligned user face data and a GUI runtime:

```bat
launchers\dfl.bat xseg editor --input-dir "workspace\data_src\aligned"
launchers\dfl.bat xseg editor --input-dir "workspace\data_dst\aligned"
```

The aligned faces are ordinary user data. Manual editing does not require Generic-XSeg pretrained weights, a pretraining dataset, or a separately supplied GUI runtime. PyQt/Qt is already part of a generated GUI variant.

### Automated Generic-XSeg application

Automated mask application is a different workflow. It requires a compatible external model directory supplied explicitly:

```bat
launchers\dfl.bat xseg apply ^
  --input-dir "<INPUT_DIR>" ^
  --model-dir "<XSEG_MODEL_DIR>"
```

The directory must contain `XSeg_256.npy`; `XSeg_data.dat` is optional. A missing or invalid resource fails actionably. There is no hidden download or developer-machine fallback.

`resources\xseg_generic_model` is an optional repository-local, user-managed convention used by the two curated generic-mask launchers. It is not a bundled/default model payload. Generic-XSeg mask application is available when a compatible model is supplied through the configurable user-managed path. Model license and artifact integrity have not established redistribution permission, so redistribution is not approved and no unofficial download source is documented here.

## Pretraining data

SAEHD and XSeg can use external pretraining data when their pretraining modes are enabled. Supply it explicitly:

```bat
launchers\dfl.bat train ^
  --model SAEHD ^
  --training-data-src-dir "<SRC_ALIGNED_DIR>" ^
  --training-data-dst-dir "<DST_ALIGNED_DIR>" ^
  --model-dir "<MODEL_DIR>" ^
  --pretraining-data-dir "<PRETRAIN_DATA_DIR>"
```

Use the current `train --help` output for the complete model-specific command surface. AMP does not currently expose the external dataset-pretraining branch.

The pretraining directory accepts exactly one of the source-supported top-level structures:

- A top-level `faceset.pak`; or
- Flat, non-recursive, lowercase `.jpg` DFL face files.

At least one usable face is required. Each usable face must carry DFL metadata, a recognized face type, and numeric, finite landmarks with exact shape `(68, 2)`. XSeg masks are not required, source/destination pairing is not part of this dataset contract, and no minimum beyond one usable face is enforced for baseline validity.

When pretraining is requested, missing or invalid data fails actionably. It does not silently switch to normal training, disable pretraining, search for a developer dataset, or download a dataset. No pretraining dataset is bundled, and redistribution of the historical validation data is not approved because dataset-specific licensing remains unverified.

## Workflow availability

The curated launchers and generic `dfl.bat` command surface provide the current packaged workflows without reproducing every historical batch filename or package layout. Primary extraction, sorting, faceset, XSeg, SAEHD/AMP/Quick96 training and merge, SAEHD/AMP export, and result-video workflows are available through those entrypoints.

The following historical package workflows require special note:

- Manual XSeg editing for source and destination aligned faces is available through the generic `xseg editor` commands documented above; duplicate fixed-path wrappers are not required.
- XSeg training with pretraining mode is available through the generic training surface when compatible external pretraining data is supplied explicitly. No dataset is bundled or downloaded.
- Quick96 training and masked merging are available through `launchers\train-quick96.bat` and `launchers\merge-quick96.bat`. Quick96 DFM export is not supported. Basic merging keeps super-resolution disabled by default and does not require FaceEnhancer.
- The historical FaceEnhancer workflow is not currently available in the packaged Torch baseline because its historical implementation depends on a backend that is not part of the packaged runtime.
- Historical in-place CPU-only package mutation is replaced by the explicit isolated CPU runtime variants.
- The third-party EbSynth launcher is not included; users manage and start their own EbSynth installation.
- Workspace reset and the historical bundled-viewer convenience workflows are not included in the current packaged baseline.

These statements describe current user-visible availability. The machine-readable historical compatibility accounting remains in `launchers\workflow_launchers.json`.

## Maintainer notes

- Keep runtime dependency changes in the source-controlled requirement lock files and canonical builder inputs.
- Preserve the four exact variant names and the packaged-interpreter-only launch boundary.
- Preserve Torch-before-PyQt initialization for XSeg Editor unless a separately validated compatibility change supersedes it.
- Treat `launchers\workflow_launchers.json` as the internal machine-readable compatibility registry and verify its ledger before changing the workflow-availability statements above.
- Keep user workspace and external model/dataset payloads out of generated runtime trees.
- Do not describe current-environment validation as release portability. Additional qualification is required before making clean-machine, moved-install, fresh-extraction, or release-package guarantees.
