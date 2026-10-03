# DeepFaceLab modernization

This branch modernizes the Windows package and execution boundary for DeepFaceLab. It uses generated, versioned Python runtimes instead of an in-place embedded environment, routes application commands through one launcher, and keeps user data and externally supplied resources outside the generated runtime.

The current package/runtime contract has been validated on the current Windows development environment and its current generated runtimes. That evidence is not a general release-portability certification. Clean-machine, different-user, moved-install, path-with-spaces, archive, fresh-extraction, and offline reconstruction scenarios are not yet qualified.

See [DISTRIBUTION.md](DISTRIBUTION.md) for the complete runtime, launcher, workspace, resource, workflow-availability, and support contract.

## Quick start

From a Windows `cmd.exe` prompt at the repository root, construct and activate one runtime variant:

```bat
launchers\dfl-setup-runtime.bat cuda-gui --activate
```

Runtime construction is a developer/build operation and requires suitable host tooling. Once a runtime is built and selected, normal application commands use only that packaged runtime's Python interpreter; they do not fall back to system Python.

Inspect the application surface and verify the selected package:

```bat
launchers\dfl.bat --help
launchers\dfl.bat --self-test
launchers\dfl-envreport.bat --verify
```

`--self-test` is a focused bootstrap diagnostic. `dfl-envreport.bat --verify` checks the current environment, external prerequisites, and runtime consistency. Neither command replaces the test suite or release-portability qualification.

## Runtime variants

| Variant | Compute | GUI workflows |
| --- | --- | --- |
| `cpu-nogui` | CPU Torch; CUDA unavailable | No |
| `cpu-gui` | CPU Torch; CUDA unavailable | Yes |
| `cuda-nogui` | CUDA-enabled Torch for supported NVIDIA environments | No |
| `cuda-gui` | CUDA-enabled Torch for supported NVIDIA environments | Yes |

Choose a GUI variant for XSeg Editor and other supported PyQt workflows. The CUDA variants require a compatible NVIDIA GPU and driver; no particular GPU model is a general requirement.

## Host prerequisites

- 64-bit Windows and the Microsoft Visual C++ 2015–2022 x64 runtime.
- `ffmpeg` and `ffprobe` available through `PATH`; they are not bundled. `PATH` order selects the executables used.
- A compatible NVIDIA driver and GPU only for `cuda-*` variants.
- Host Python/build tooling only when constructing runtimes, not for normal application launch.

Historical avecl/OpenCL setup is not a prerequisite of the current production runtime contract.

## Workspace and external resources

The workspace is user-managed data. Launchers commonly use:

```text
workspace\data_src
workspace\data_dst
workspace\model
```

Runtime construction does not package faces, source or destination media, checkpoints, previews, training history, or other workspace content. Individual workflows may create output or model directories when invoked.

Some features require resources that are deliberately not bundled:

- Automated Generic-XSeg mask application requires a compatible user-supplied model directory.
- SAEHD and XSeg pretraining, when enabled, require a user-supplied pretraining dataset.

Manual XSeg mask editing is different: it needs aligned user face data and a GUI runtime, but it does not need Generic-XSeg weights or a pretraining dataset.

## Workflow availability

The current packaged surface covers the primary extraction, sorting, faceset, XSeg, SAEHD/AMP training, merge, export, and result-video workflows. Some historical package workflows have different availability:

- Quick96 training and merging are not included in the current packaged baseline.
- The historical FaceEnhancer workflow is not currently available in the packaged Torch baseline.
- XSeg training with pretraining mode requires explicitly supplied external pretraining data.
- Third-party and convenience applications such as EbSynth and bundled viewers are not included.

See [DISTRIBUTION.md](DISTRIBUTION.md#workflow-availability) for the detailed current workflow boundary.
