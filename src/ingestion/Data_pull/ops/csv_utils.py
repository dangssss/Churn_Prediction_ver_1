# ops/csv_utils.py
from pathlib import Path
import csv
import io


def sniff_delimiter_from_head(file_path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """
    Đoán delimiter từ phần đầu file (',', ';', '\\t', '|').
    """
    with open(file_path, "rb") as f:
        head = f.read(sample_bytes)
    txt = head.decode("utf-8", errors="ignore")
    cands = [",", ";", "\t", "|"]
    best, score = ",", -1
    lines = txt.splitlines()[:1000]

    for d in cands:
        cnts = [ln.count(d) for ln in lines if ln.strip()]
        if not cnts:
            continue
        s = min(cnts)
        if s > score:
            best, score = d, s

    return best


class GeneratorFile:
    """
    Adapter biến generator(str) thành file-like cho psycopg2.copy_expert().
    """
    def __init__(self, gen):
        self.gen = gen
        self.buffer = ""

    def read(self, size: int = -1) -> str:
        if size is None or size < 0:
            chunks = [self.buffer]
            self.buffer = ""
            for part in self.gen:
                chunks.append(part)
            return "".join(chunks)

        while len(self.buffer) < size:
            try:
                self.buffer += next(self.gen)
            except StopIteration:
                break

        out = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return out


def csv_stream_canonical(
    file_path: Path,
    detected_delim: str,
    expected_header: list[str],
    batch_rows: int = 50_000,
    source_has_header: bool = True,
    injection_mode: str = "sanitize",
    text_cols: set[str] | None = None,
):
    """
    Stream CSV -> canonical (UTF-8, ',', '"'), chèn header nếu nguồn không có header.
    Guard injection ở cột text: off | text_only | strict | sanitize
    """
    text_cols = text_cols or set()

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=detected_delim, quotechar='"')
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=",", quotechar='"', lineterminator="\n")

        def sanitize_row(row, text_idx, mode):
            if mode == "off":
                return row

            danger = ("=", "@") if mode in ("text_only", "sanitize") else ("=", "+", "-", "@")
            out = list(row)

            for i in text_idx:
                v = out[i]
                if isinstance(v, str) and len(v) > 0 and v[0] in danger:
                    if mode == "sanitize":
                        out[i] = "'" + v
                    else:
                        raise ValueError("CSV injection-like cell detected (text column)")
            return out

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("Empty CSV")

        # 1) Nguồn đã có header
        if source_has_header:
            header = first_row
            if header != expected_header:
                raise ValueError(f"Header mismatch: got {header} vs {expected_header}")
            num_cols = len(header)
            text_idx = [i for i, c in enumerate(header) if c in text_cols]
            writer.writerow(header)
        # 2) Nguồn không có header → chèn expected_header
        else:
            header = expected_header
            num_cols = len(header)
            text_idx = [i for i, c in enumerate(header) if c in text_cols]
            writer.writerow(header)

            if len(first_row) != num_cols:
                raise ValueError(f"Ragged row {len(first_row)} cols (expect {num_cols})")

            first_row = sanitize_row(first_row, text_idx, injection_mode)
            writer.writerow(first_row)

        # flush batch đầu tiên (header + có thể 1 dòng data)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        batch = 0
        for row in reader:
            if len(row) != num_cols:
                raise ValueError(f"Ragged row {len(row)} cols (expect {num_cols})")
            row = sanitize_row(row, text_idx, injection_mode)
            writer.writerow(row)
            batch += 1

            if batch >= batch_rows:
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)
                batch = 0

        if batch > 0:
            yield buf.getvalue()
