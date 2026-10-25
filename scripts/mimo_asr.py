"""调用小米 MiMo ASR 转写音频（本地密钥，DPAPI 加密存储）。

密钥读取顺序：
1. 环境变量 MIMO_API_KEY
2. .secrets/xiaomi_api_key.dpapi（由 PowerShell ConvertFrom-SecureString 生成，绑定当前 Windows 用户）

用法：
    .\\venv\\Scripts\\python.exe scripts\\mimo_asr.py --audio "a.mp3" --out local\\a.md
"""
import argparse
import base64
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-asr"
MAX_B64_BYTES = 10 * 1024 * 1024  # 接口限制：base64 编码后 <= 10 MB


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _unprotect_dpapi(hex_blob: str) -> str:
    blob = bytes.fromhex(hex_blob.strip())
    buf_in = ctypes.create_string_buffer(blob, len(blob))
    din = _DATA_BLOB(len(blob), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_char)))
    dout = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(din), None, None, None, None, 0, ctypes.byref(dout)
    ):
        raise OSError("CryptUnprotectData 失败：密钥文件不属于当前 Windows 用户")
    try:
        raw = ctypes.string_at(dout.pbData, dout.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dout.pbData)
    return raw.decode("utf-16-le")


def load_api_key() -> str:
    key = os.environ.get("MIMO_API_KEY")
    if key:
        return key.strip()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".secrets", "xiaomi_api_key.dpapi")
    if not os.path.exists(path):
        sys.exit(f"未找到密钥：{path}，或设置环境变量 MIMO_API_KEY")
    with open(path, "r", encoding="ascii") as f:
        return _unprotect_dpapi(f.read())


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def make_chunks(src: str, out_dir: str, start: float, dur: float, idx: int) -> str:
    """切出一段并转 16 kHz 单声道 48 kbps，控制 base64 体积"""
    dst = os.path.join(out_dir, f"chunk_{idx:02d}.mp3")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
         "-ac", "1", "-ar", "16000", "-b:a", "48k", dst],
        check=True,
    )
    return dst


def transcribe(path: str, api_key: str, language: str = "zh", retries: int = 3) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    if len(b64) > MAX_B64_BYTES:
        raise ValueError(f"{path} base64 后 {len(b64)/1048576:.1f} MB，超过 10 MB 上限")
    payload = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": f"data:audio/mpeg;base64,{b64}"},
            }],
        }],
        "asr_options": {"language": language},
    }).encode("utf-8")

    last_err = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API_URL, data=payload,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                print("  ! 输出被长度截断，建议减小 --chunk-seconds", file=sys.stderr)
            return choice["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:  # 网络抖动
            last_err = repr(e)
        print(f"  重试 {attempt}/{retries}：{last_err}", file=sys.stderr)
        time.sleep(2 * attempt)
    raise RuntimeError(f"转写失败：{last_err}")


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-seconds", type=float, default=240.0)
    ap.add_argument("--language", default="zh", choices=["auto", "zh", "en"])
    ap.add_argument("--keep-chunks", action="store_true")
    args = ap.parse_args()

    api_key = load_api_key()
    total = probe_duration(args.audio)
    print(f"音频时长 {fmt_ts(total)}，切片 {args.chunk_seconds:.0f}s")

    tmp = tempfile.mkdtemp(prefix="mimo_asr_")
    parts = []
    try:
        start, idx = 0.0, 0
        while start < total - 1.0:
            dur = min(args.chunk_seconds, total - start)
            chunk = make_chunks(args.audio, tmp, start, dur, idx)
            size = os.path.getsize(chunk) / 1048576
            print(f"[{idx}] {fmt_ts(start)} +{dur:.0f}s  {size:.2f} MB")
            text = transcribe(chunk, api_key, args.language)
            parts.append((start, text))
            print(f"    -> {len(text)} 字")
            start += dur
            idx += 1
    finally:
        if args.keep_chunks:
            print(f"切片保留在 {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# 转写：{os.path.basename(args.audio)}\n\n")
        for start, text in parts:
            f.write(f"## [{fmt_ts(start)}]\n\n{text.strip()}\n\n")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
