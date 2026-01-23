from pathlib import Path
import pycurl
import math
import zipfile
import shutil
from tqdm import tqdm

URL = "https://zenodo.org/records/14169410/files/data_NeuroConText.zip"
# Save the zip in the current folder temporarily
ZIP_FILE = Path("data_NeuroConText.zip")
# Final destination for extracted folders
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONNECTIONS = 8 # Increased for faster download on high-speed lines

def get_size(url: str) -> int:
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url); c.setopt(pycurl.NOBODY, True)
    c.setopt(pycurl.FOLLOWLOCATION, True); c.perform()
    size = int(c.getinfo(pycurl.CONTENT_LENGTH_DOWNLOAD))
    c.close()
    return size

def download_parallel(url: str, out: Path, n: int):
    total = get_size(url)
    chunk = math.ceil(total / n)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    f = open(out, "wb")
    f.truncate(total)
    
    m = pycurl.CurlMulti()
    curls = []
    bar = tqdm(total=total, unit="B", unit_scale=True, desc="[1/2] Downloading", colour="green")

    def write_factory(start):
        offset = start
        def write(data):
            nonlocal offset
            f.seek(offset); f.write(data)
            offset += len(data); bar.update(len(data))
            return len(data)
        return write

    for i in range(n):
        start = i * chunk
        end = min(start + chunk - 1, total - 1)
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url); c.setopt(pycurl.RANGE, f"{start}-{end}")
        c.setopt(pycurl.WRITEFUNCTION, write_factory(start))
        c.setopt(pycurl.FOLLOWLOCATION, True); c.setopt(pycurl.NOSIGNAL, 1)
        curls.append(c); m.add_handle(c)

    # Event Loop
    while True:
        ret, active = m.perform()
        if ret != pycurl.E_CALL_MULTI_PERFORM: break
    while active:
        m.select(1.0)
        while True:
            ret, active = m.perform()
            if ret != pycurl.E_CALL_MULTI_PERFORM: break

    for c in curls: m.remove_handle(c); c.close()
    bar.close(); f.close()

def extract_and_cleanup(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\033[94m[2/2] Extracting to {target_dir}...\033[0m")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        # Progress bar for extraction
        for member in tqdm(members, desc="Unzipping  ", colour="cyan"):
            # Extract to target_dir (../data)
            zip_ref.extract(member, target_dir)
            
    # Remove the zip file after successful extraction
    zip_path.unlink()
    print(f"\033[92m✔ Setup Complete. Data ready in {target_dir}\033[0m")

if __name__ == "__main__":
    # 1. Download
    download_parallel(URL, ZIP_FILE, CONNECTIONS)
    # 2. Unzip and move to ../data
    extract_and_cleanup(ZIP_FILE, DATA_DIR)