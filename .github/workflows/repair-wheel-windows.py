#!/usr/bin/env python
import zipfile
import os
import shutil


def write_dll(wheel_path, dlls):
    """Insert DLLs into a wheel"""
    print(f"Inserting dlls into wheel: {wheel_path}")
    with zipfile.ZipFile(wheel_path, mode="a") as wheel:
        for local_dll in dlls:
            name = os.path.basename(local_dll)
            archive_path = os.path.join("pywr", ".libs", name)
            print(f'Writing local dll "{local_dll}" to archive path "{archive_path}".')
            wheel.write(local_dll, archive_path)
    print("Wheel repaired!")


def copy_wheel(original, dest_dir):
    """Copy wheel to dest_dir"""
    wheel_name = os.path.basename(original)
    new_wheel = os.path.join(dest_dir, wheel_name)
    print(f'Copying wheel from "{wheel_name}" to "{new_wheel}"')
    shutil.copy(original, new_wheel)
    return new_wheel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=str)
    parser.add_argument("dest_dir", type=str)
    parser.add_argument("glpk_dll", type=str)
    parser.add_argument("lpsolve_dll", type=str)
    args = parser.parse_args()

    new_wheel_path = copy_wheel(args.wheel, args.dest_dir)
    write_dll(new_wheel_path, [args.glpk_dll, args.lpsolve_dll])
