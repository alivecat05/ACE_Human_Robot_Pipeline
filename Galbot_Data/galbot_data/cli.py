"""Simple MCAP processing entry point: -i input -o output."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def discover_recordings(value):
    path = Path(value).resolve()
    if path.is_file() and path.suffix.lower() == ".mcap":
        files = [path]
    elif path.is_dir():
        files = sorted({p.resolve() for p in path.rglob("*")
                        if p.is_file() and p.suffix.lower() == ".mcap"})
    else:
        raise ValueError(f"不是 MCAP 文件或目录: {path}")
    if not files:
        raise ValueError(f"输入目录及其子目录中没有 .mcap 文件: {path}")
    names = {}
    for source in files:
        if source.stem in names:
            raise ValueError(f"录制名称冲突，请分开输出或重命名文件: {names[source.stem]} 与 {source}")
        names[source.stem] = source
    return files


def process_one(source, root, *, train=False, head="left"):
    from .extract import extract_recording

    destination = root / source.stem
    if destination.exists():
        raise FileExistsError(f"录制输出已存在，请选择新输出目录: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.stem}-", dir=root))
    try:
        manifest = extract_recording(source, temporary, train=train, head=head)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_extract(source, output, *, train=False, head="left"):
    sources = discover_recordings(source)
    root = Path(output).resolve()
    for path in sources:
        if (root / path.stem).exists():
            raise FileExistsError(f"输出已存在，请选择新输出目录: {root / path.stem}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(sources)} 个 MCAP，输出到 {root}", flush=True)
    failures = 0
    for path in sources:
        try:
            manifest = process_one(path, root, train=train, head=head)
            frames = sum(camera["frames"] for camera in manifest["cameras"].values())
            print(f"完成 {path.name}: {manifest['action_rows']} 行关节数据，{frames} 帧视频", flush=True)
            if manifest["missing_camera_topics"]:
                print(f"  缺少相机 topic: {', '.join(manifest['missing_camera_topics'])}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"失败 {path}: {exc}", file=sys.stderr, flush=True)
    if failures:
        print(f"{failures} 个文件失败；其他成功的录制已保留。", file=sys.stderr)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="提取 MCAP，每段录制的 action.csv 和四路视频放在同一文件夹")
    parser.add_argument("-i", "--input", type=Path, required=True, help="MCAP 文件或目录（自动搜索子目录）")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--train", action="store_true", help="只输出头部单目、左右臂三路视频和 action.csv")
    parser.add_argument("--head", choices=("left", "right"), default="left",
                        help="--train 模式使用的头部相机，默认 left；普通模式仍输出双目")
    args = parser.parse_args(argv)
    try:
        return run_extract(args.input, args.output, train=args.train, head=args.head)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
