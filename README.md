# Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures

**ReSYNC · CoRL 2026**

[Bowen Li](https://jaraxxus-me.github.io/), [Mayank Mishra](https://mmayank74567.github.io/), [Y. Isabel Liu](https://yil42.github.io/), [Stone Tao](https://stonet2000.github.io/), [Nishanth Kumar](https://nishanthjkumar.com/), Alexander Gray, Ruwan Wickramarachchi, [Jonathan Francis](https://jonfranc.com/), [Sebastian Scherer](http://theairlab.org/team/sebastian/), and [Tom Silver](https://tomsilver.github.io/).

[Paper](https://arxiv.org/abs/2606.18328) · [Project website](https://jaraxxus-me.github.io/ReSYNC/) · [release assets](https://drive.google.com/drive/folders/1w1gB9M4SrqF-3-vQY6t6R-TdOFs4ycID)

## Abstract

Intelligent robots should not only recover from failures, but also acquire the abstract knowledge needed to avoid them in the future. While reinforcement learning (RL) can learn reactive recovery behaviors, training a separate policy for every distinct failure mode is highly inefficient. We introduce Recovery-Driven Synthesis of Relational Concepts (ReSYNC), the first approach that progressively discovers and refines state abstractions (relational predicates) from failure-recovery experience to support abstract planning. Unlike purely reactive methods, ReSYNC jointly learns skills and concepts through an incremental dual-learning process. In the skill-learning phase, the robot uses RL to learn to recover from failures seen in training tasks. In the concept-learning phase, the robot discovers new relational predicates and refines its abstract planning model to explain and generalize the learned recovery behaviors. This interaction enables ReSYNC to convert local recoveries seen during training into global failure avoidance at test time. Across four simulated domains, we show that ReSYNC's ability to continually expand and refine its abstraction library allows it to solve long-horizon, previously unseen problems, outperforming strong baselines by over 50%. Additionally, we demonstrate sim-to-real transfer of ReSYNC, where it performs real-world non-prehensile manipulation skills and generalizes to unseen scenarios through abstract planning. Overall, ReSYNC represents a significant step toward robots that autonomously acquire abstractions for scalable, failure-aware planning in the physical world.

## Source snapshot

This initial release copies the tracked `main` tree of `Jaraxxus-Me/skill_refactor` at **`49f8d12a072166e89ef45a0eacc1d20adc8bdefa`**, with no imported Git history. The original experiment source, configurations, tests, and scripts are preserved. Simulation assets are distributed through Google Drive and restored to their original paths by the downloader. `prpl-mono` is vendored at its pinned commit **`27f6770190ccdd7758d238afe9f989fc13889e9f`**; no submodule checkout is required.

Publication and testing documentation and release ignore rules are adapted for this release. Release scripts are additions alongside the unchanged research code. The upstream preparation guide is retained in [release/UPSTREAM_README.md](release/UPSTREAM_README.md).

```bash
python scripts/verify_source_snapshot.py
```

Run this after downloading simulation assets below. This checks 1,381 imported files against their original Git blob hashes, including the vendored dependency.

## Installation

Use Linux with an NVIDIA GPU, working NVIDIA graphics/Vulkan drivers, and Python 3.11. The release is tested on an RTX 3090 Ti with NVIDIA driver 580.173.02. Cluttered Drawer uses GPU PhysX; the original configurations also run neural inference on CUDA.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this repository's root:

```bash
git clone https://github.com/Jaraxxus-Me/ReSYNC.git
cd ReSYNC
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r release/requirements-linux.lock
uv pip install --no-deps -e . \
  -e third-party/prpl-mono/prpl-utils \
  -e third-party/prpl-mono/prbench \
  -e third-party/prpl-mono/relational-structs \
  -e third-party/prpl-mono/toms-geoms-2d
```

The lock file records the tested Linux environment, including PyTorch 2.11.0, NumPy 1.26.4, ManiSkill 3.0.0b21, SAPIEN 3.0.3, and Pymunk 7.2.0. Run all commands below in this environment, from the repository root.

## Asset preparation

The [release folder](https://drive.google.com/drive/folders/1w1gB9M4SrqF-3-vQY6t6R-TdOFs4ycID) contains the trained policy checkpoints, predicate and terminal networks, effect vectors, variable bindings, symbolic operator JSON files, and supporting datasets. Downloads are split into parts of at most 90 MiB. The downloader reassembles each ZIP and verifies SHA-256 checksums for both archives and extracted files using [release/assets_manifest.json](release/assets_manifest.json).

For the two installation sanity checks:

```bash
python scripts/download_release_assets.py --domains simulation_assets blocked_stacking cluttered_drawer
```

To obtain the simulation assets and all four simulated-domain model bundles, including meshes and object models:

```bash
python scripts/download_release_assets.py
```

| Bundle | Contents | Compressed size |
| --- | --- | --- |
| Simulation assets | Robot, drawer, hammer, YCB objects, and other meshes | 90 MiB |
| Blocked Stacking | Punch, Wiggle, Fiddle policies; stage-1/2/3 predicates; supporting data | 206 MiB |
| Cluttered Drawer | Recovery policies and available predicate bundles; supporting data | 317 MiB |
| Icy Transport | IcyDrive and MuddyDrive policies; stage-1/2 predicates; supporting data | 92 MiB |
| Cluttered Room (Rearrange) | Push policy, predicates, and supporting data | 28 MiB |

The archives preserve the saved files, including intermediate policy checkpoints. The drawer `sc123` predicate folder contains the available stage-1/2 metadata; it is not a completed stage-3 reproduction. Initial-release inference is verified for stage 1 of Blocked Stacking and Cluttered Drawer. The other artifacts are provided for subsequent reproduction work.

Planner datasets remain necessary even for inference: the original loader reconstructs candidate predicates and reads training plan skeletons before loading the saved operators. The release uses the original full-method artifacts.

The simulation-assets bundle includes the ManiSkill YCB objects used by Cluttered Drawer. **Set `MS_ASSET_DIR` before importing ManiSkill**, including in every new shell:

```bash
export MS_ASSET_DIR="$PWD/src/skill_refactor/assets/maniskill_assets"
```

The robot, drawer, hammer, YCB, and other meshes are restored under `src/skill_refactor/assets/`, which is excluded from Git. ReplicaCAD and the MS-HAB datasets are only needed for the future Rearrange experiments; their upstream preparation instructions remain in [release/UPSTREAM_README.md](release/UPSTREAM_README.md).

## Installation sanity checks: learned skills and predicates

Run the following commands after installation and asset preparation:

```bash
python scripts/sanity_check.py --env blocked_stacking --episodes 5
python scripts/sanity_check.py --env cluttered_drawer --episodes 5
```

Each command loads the original policy, neural predicates, and symbolic operators, then evaluates both configurations:

- **`1_g`**: the obstruction prevents grasping, as in the recovery training distribution.
- **`1_b`**: the obstruction is at the placement/stacking target, testing reuse of the learned recovery in a new composition.

The check uses `LifelongRefApproach` and the original stage-1 configuration, without training. It follows the upstream evaluation's reset schedule, action conversion, and stopping conditions, and omits mandatory video recording. It writes per-episode outcomes and executed operator sequences to `results/sanity_<environment>.json`, with detailed logs beside that file. It exits unsuccessfully if either configuration never succeeds or never executes its learned recovery (`Punch` or `Pull`). Individual failed episodes are expected with the learned policies.

For 50 episodes per configuration:

```bash
python scripts/sanity_check.py --env blocked_stacking --episodes 50
python scripts/sanity_check.py --env cluttered_drawer --episodes 50
```

These are installation checks rather than a full reproduction of the paper's aggregate results. The source YAML settings are preserved, including the stage-1 rollout limits (300 for Blocked Stacking and 900 for Cluttered Drawer).

## Release validation

The installation commands were tested in a second clean Python 3.11 environment on October 5, 2026. The five asset bundles were downloaded from Drive without authentication and verified: 2,856 files, 11 archive parts. All 1,381 preserved source files matched the upstream Git blob hashes.

| Environment | `1_b` successes | `1_g` successes |
| --- | --- | --- |
| Blocked Stacking | 4/5 | 2/5 |
| Cluttered Drawer | 4/5 | 5/5 |

Machine-readable installation, public-access, and per-episode inference records are in [release/validation](release/validation/). The sanity checks exercise learned recovery actions in both configurations. These small samples validate installation and inference; they do not estimate the paper's aggregate performance. GPU simulation can produce variation between runs even with the same reset seeds.

## Citation

```bibtex
@inproceedings{li2026recover,
  title={Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures},
  author={Li, Bowen and Mishra, Mayank and Liu, Y. Isabel and Tao, Stone and Kumar, Nishanth and Gray, Alexander and Wickramarachchi, Ruwan and Francis, Jonathan and Scherer, Sebastian and Silver, Tom},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```

The source retains its upstream [MIT license](LICENSE) and dependency licenses.

## RoadMap

- [x] Initial release, installation, and sanity checks.
- [ ] Test predicate learning in BlockedStacking stage 1.
- [ ] Incorporate the IsaacLab FlippedStacking environment: inference with learned predicates and skills.
