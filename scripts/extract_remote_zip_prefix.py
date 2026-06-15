import argparse
import pathlib
import subprocess
import struct
import zlib


EOCD = b"PK\x05\x06"
ZIP64_EOCD = b"PK\x06\x06"
CD_FILE_HEADER = b"PK\x01\x02"
LOCAL_FILE_HEADER = b"PK\x03\x04"


def curl_bytes(args: list[str]) -> bytes:
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return result.stdout


def range_get(url: str, start: int, end: int) -> bytes:
    return curl_bytes(
        [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-r",
            f"{start}-{end}",
            url,
        ]
    )


def parse_zip64_extra(extra: bytes, need_uncomp: bool, need_comp: bool, need_offset: bool):
    pos = 0
    values = {}
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        data = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        cursor = 0
        if need_uncomp:
            values["uncomp_size"] = struct.unpack_from("<Q", data, cursor)[0]
            cursor += 8
        if need_comp:
            values["comp_size"] = struct.unpack_from("<Q", data, cursor)[0]
            cursor += 8
        if need_offset:
            values["local_offset"] = struct.unpack_from("<Q", data, cursor)[0]
    return values


def central_directory(url: str, size: int):
    tail_size = min(size, 65536)
    tail = range_get(url, size - tail_size, size - 1)
    eocd_pos = tail.rfind(EOCD)
    if eocd_pos < 0:
        raise RuntimeError("Could not find ZIP EOCD")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
    cd_size = eocd[5]
    cd_offset = eocd[6]
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        zip64_pos = tail.rfind(ZIP64_EOCD, 0, eocd_pos)
        if zip64_pos < 0:
            raise RuntimeError("ZIP64 EOCD missing")
        zip64 = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_pos)
        cd_size = zip64[8]
        cd_offset = zip64[9]
    return range_get(url, cd_offset, cd_offset + cd_size - 1)


def parse_central_directory(cd: bytes):
    entries = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != CD_FILE_HEADER:
            raise RuntimeError(f"Bad central directory signature at {pos}")
        fields = struct.unpack_from("<4s6H3L5H2L", cd, pos)
        method = fields[4]
        crc = fields[7]
        comp_size = fields[8]
        uncomp_size = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        local_offset = fields[16]
        name_start = pos + 46
        name = cd[name_start : name_start + name_len].decode("utf-8")
        extra = cd[name_start + name_len : name_start + name_len + extra_len]
        need_uncomp = uncomp_size == 0xFFFFFFFF
        need_comp = comp_size == 0xFFFFFFFF
        need_offset = local_offset == 0xFFFFFFFF
        zip64 = parse_zip64_extra(extra, need_uncomp, need_comp, need_offset)
        uncomp_size = zip64.get("uncomp_size", uncomp_size)
        comp_size = zip64.get("comp_size", comp_size)
        local_offset = zip64.get("local_offset", local_offset)
        entries.append(
            {
                "name": name,
                "method": method,
                "crc": crc,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "local_offset": local_offset,
            }
        )
        pos = name_start + name_len + extra_len + comment_len
    return entries


def extract_entry(url: str, entry: dict, out_root: pathlib.Path):
    name = entry["name"]
    if name.endswith("/"):
        return
    header = range_get(url, entry["local_offset"], entry["local_offset"] + 29)
    if header[:4] != LOCAL_FILE_HEADER:
        raise RuntimeError(f"Bad local header for {name}")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len = fields[9]
    extra_len = fields[10]
    data_start = entry["local_offset"] + 30 + name_len + extra_len
    comp = range_get(url, data_start, data_start + entry["comp_size"] - 1)
    if entry["method"] == 0:
        data = comp
    elif entry["method"] == 8:
        data = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {entry['method']} for {name}")
    if len(data) != entry["uncomp_size"]:
        raise RuntimeError(f"Size mismatch for {name}: {len(data)} != {entry['uncomp_size']}")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if crc != entry["crc"]:
        raise RuntimeError(f"CRC mismatch for {name}")
    out_path = out_root / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    headers = curl_bytes(
        [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "-I",
            args.url,
        ]
    ).decode("latin1")
    size = None
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
    if size is None:
        raise RuntimeError("Content-Length missing")
    cd = central_directory(args.url, size)
    entries = [entry for entry in parse_central_directory(cd) if entry["name"].startswith(args.prefix)]
    total = sum(entry["uncomp_size"] for entry in entries if not entry["name"].endswith("/"))
    print(f"matched {len(entries)} entries under {args.prefix}, uncompressed={total / 1024**3:.2f} GiB")
    for index, entry in enumerate(entries, 1):
        if entry["name"].endswith("/"):
            continue
        print(f"[{index}/{len(entries)}] {entry['name']} ({entry['uncomp_size'] / 1024**2:.1f} MiB)")
        extract_entry(args.url, entry, args.out_dir)


if __name__ == "__main__":
    main()
